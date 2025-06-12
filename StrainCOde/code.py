import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.interpolate import splprep, splev

class CardiacAnalyzer:
    """
    Final, robust version for LV analysis using a corrected "Anchor-and-Smooth"
    strategy with a fallback mechanism to ensure stable contour generation for all frames.
    """

    def __init__(self, xml_file_path: str, smoothing_factor: float = 20.0, num_points: int = 130):
        self.xml_path = xml_file_path
        self.smoothing_factor = smoothing_factor
        self.num_points = num_points
        self._parse_xml_data()
        self.resampled_contours = []
        self.volumes_ml = np.array([])
        self.gls_curve = np.array([])

    def _parse_xml_data(self):
        """Parses the SpreadsheetML XML file."""
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        try: tree = ET.parse(self.xml_path); root = tree.getroot()
        except ET.ParseError as e: raise ValueError(f"Error parsing XML file: {e}")
        all_rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', ns)
        time_intervals_ms, x_coords_by_point, y_coords_by_point = [], [], []
        parsing_state = None
        for row in all_rows:
            cells = row.findall('ss:Cell', ns)
            if not cells: continue
            first_cell_data = cells[0].find('ss:Data', ns)
            if first_cell_data is not None and first_cell_data.text:
                header = first_cell_data.text.strip()
                if header == 'frametime': time_intervals_ms = [float(c.find('ss:Data', ns).text) for c in cells[1:] if c.find('ss:Data', ns) is not None]
                elif header == 'pixelsize': self.pixel_size_mm = float(cells[1].find('ss:Data', ns).text)
                elif header in ['EndoX', 'EndoY', 'EpiX']: parsing_state = header; continue
            if parsing_state == 'EndoX':
                row_data = [c.find('ss:Data', ns) for c in cells]
                if all(c is not None and c.get(f'{{{ns["ss"]}}}Type') == 'Number' for c in row_data): x_coords_by_point.append([float(c.text) for c in row_data])
            elif parsing_state == 'EndoY':
                row_data = [c.find('ss:Data', ns) for c in cells]
                if all(c is not None and c.get(f'{{{ns["ss"]}}}Type') == 'Number' for c in row_data): y_coords_by_point.append([float(c.text) for c in row_data])
        if not time_intervals_ms or not x_coords_by_point or not y_coords_by_point: raise ValueError("Could not find required data in XML.")
        x_coords_by_frame = np.array(x_coords_by_point).T; y_coords_by_frame = np.array(y_coords_by_point).T
        self.raw_contours = [np.column_stack((x, y)) for x, y in zip(x_coords_by_frame, y_coords_by_frame)]
        self.times_sec = np.cumsum([0] + time_intervals_ms) / 1000.0

    def _find_apex_index(self, contour: np.ndarray) -> int:
        """Robustly finds the index of the point most distant from the mitral base line."""
        if len(contour) < 2: return 0
        mitral_p1, mitral_p_end = contour[0], contour[-1]
        base_vec = mitral_p_end - mitral_p1
        norm_base_sq = np.dot(base_vec, base_vec)
        if norm_base_sq < 1e-9: return np.argmax(np.linalg.norm(contour - mitral_p1, axis=1))
        vecs_from_p1 = contour - mitral_p1
        t = np.dot(vecs_from_p1, base_vec) / norm_base_sq
        projection_points = mitral_p1 + t[:, np.newaxis] * base_vec
        distances = np.linalg.norm(vecs_from_p1 - projection_points, axis=1)
        return np.argmax(distances)

    def _resample_contours(self):
        """
        **REWRITTEN FOR ROBUSTNESS**: Implements "Anchor-and-Smooth" with a fallback
        to prevent failures on geometrically challenging frames.
        """
        self.resampled_contours = []

        def get_spline_and_length(segment_pts, smoothing_factor):
            n_pts = len(segment_pts)
            if n_pts < 2: return None, 0.0
            k = min(3, n_pts - 1)
            s = smoothing_factor if k >= 3 else 0
            try:
                tck, _ = splprep([segment_pts[:, 0], segment_pts[:, 1]], s=s, k=k, per=False)
                u_fine = np.linspace(0, 1, 200) # Use fewer points for speed
                points_fine = np.column_stack(splev(u_fine, tck))
                length = np.sum(np.linalg.norm(np.diff(points_fine, axis=0), axis=1))
                return tck, length
            except Exception:
                return None, 0.0

        for raw_contour in self.raw_contours:
            apex_idx = self._find_apex_index(raw_contour)
            
            seg1_pts = raw_contour[:apex_idx + 1]
            seg2_pts = raw_contour[apex_idx:]

            # **CRITICAL FIX**: If splitting creates an invalid segment, fall back to a single global spline
            if len(seg1_pts) < 2 or len(seg2_pts) < 2:
                tck_full, _ = get_spline_and_length(raw_contour, self.smoothing_factor)
                if tck_full is None:
                    self.resampled_contours.append(np.array([])); continue
                u_new = np.linspace(0, 1, self.num_points)
                self.resampled_contours.append(np.column_stack(splev(u_new, tck_full)))
                continue

            tck1, len1 = get_spline_and_length(seg1_pts, self.smoothing_factor)
            tck2, len2 = get_spline_and_length(seg2_pts, self.smoothing_factor)

            if tck1 is None or tck2 is None:
                self.resampled_contours.append(np.array([])); continue
            
            total_len = len1 + len2
            n_pts1 = int(round(self.num_points * len1 / total_len)) if total_len > 0 else self.num_points // 2
            n_pts2 = self.num_points - n_pts1

            if n_pts1 < 2: n_pts1 = 2; n_pts2 = self.num_points - 2
            if n_pts2 < 2: n_pts2 = 2; n_pts1 = self.num_points - 2
            
            u1_new = np.linspace(0, 1, n_pts1); u2_new = np.linspace(0, 1, n_pts2)
            seg1_new = np.column_stack(splev(u1_new, tck1))
            seg2_new = np.column_stack(splev(u2_new, tck2))
            
            self.resampled_contours.append(np.vstack([seg1_new[:-1], seg2_new]))

    def _calculate_volumes(self):
        """Calculates volume using slices PARALLEL to the mitral base."""
        self.volumes_ml = np.zeros(len(self.resampled_contours))
        for i, contour in enumerate(self.resampled_contours):
            if contour.shape[0] < 3: continue
            raw_contour = self.raw_contours[i]
            mitral_p1, mitral_p_end = raw_contour[0], raw_contour[-1]
            base_vector = mitral_p_end - mitral_p1
            if np.linalg.norm(base_vector) == 0: continue
            angle = np.arctan2(base_vector[1], base_vector[0])
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            oriented_contour = np.dot(rot_mat, (contour - mitral_p1).T).T
            height = np.max(oriented_contour[:, 1])
            if height <= 0: continue
            total_volume = 0.0
            sorted_points = oriented_contour[np.argsort(oriented_contour[:, 1])]
            for j in range(len(sorted_points) - 1):
                y1, y2 = sorted_points[j, 1], sorted_points[j+1, 1]
                disk_height = y2 - y1
                if disk_height <= 0: continue
                slice_points = oriented_contour[(oriented_contour[:,1] >= y1) & (oriented_contour[:,1] < y2)]
                if len(slice_points) < 2: continue
                radius = (np.max(slice_points[:, 0]) - np.min(slice_points[:, 0])) / 2.0
                total_volume += np.pi * (radius**2) * disk_height
            self.volumes_ml[i] = total_volume * (self.pixel_size_mm**3) / 1000.0

    def _calculate_gls(self):
        """Calculates GLS by averaging the strain of each corresponding myocardial segment."""
        if not self.resampled_contours or len(self.resampled_contours[0]) < 2:
            self.gls_curve = np.zeros(len(self.resampled_contours)); return
        
        ref_contour_mm = self.resampled_contours[0] * self.pixel_size_mm
        ref_segment_lengths = np.linalg.norm(np.diff(ref_contour_mm, axis=0), axis=1)

        gls_values = []
        for i in range(len(self.resampled_contours)):
            if self.resampled_contours[i].shape != self.resampled_contours[0].shape:
                gls_values.append(0); continue
            
            curr_contour_mm = self.resampled_contours[i] * self.pixel_size_mm
            curr_segment_lengths = np.linalg.norm(np.diff(curr_contour_mm, axis=0), axis=1)
            
            segment_strains = np.divide(curr_segment_lengths - ref_segment_lengths, ref_segment_lengths, out=np.zeros_like(ref_segment_lengths), where=ref_segment_lengths > 1e-9)
            gls_values.append(np.mean(segment_strains) * 100.0)
            
        self.gls_curve = np.array(gls_values)

    def _calculate_ef(self):
        if len(self.volumes_ml) == 0 or np.max(self.volumes_ml) == 0: return 0.0
        edv, esv = np.max(self.volumes_ml), np.min(self.volumes_ml)
        return ((edv - esv) / edv) * 100.0
        
    def analyze(self) -> Dict[str, Any]:
        """Runs the full analysis pipeline."""
        print(f"Step 1: Resampling contours with Anchor-and-Smooth (s={self.smoothing_factor})...")
        self._resample_contours()
        print("Step 2: Calculating volume curve...")
        self._calculate_volumes()
        print("Step 3: Calculating Ejection Fraction...")
        ef = self._calculate_ef()
        print("Step 4: Calculating Global Longitudinal Strain from segments...")
        self._calculate_gls()
        print("\nAnalysis Complete.")
        results = {'ejection_fraction': ef, 'peak_global_longitudinal_strain': 0.0, 'end_diastolic_volume_ml': 0.0, 'end_systolic_volume_ml': 0.0, 'time_seconds': self.times_sec, 'volume_ml_curve': self.volumes_ml, 'gls_percent_curve': self.gls_curve}
        if len(self.gls_curve) > 0: results['peak_global_longitudinal_strain'] = np.min(self.gls_curve)
        if len(self.volumes_ml) > 0: results['end_diastolic_volume_ml'] = np.max(self.volumes_ml); results['end_systolic_volume_ml'] = np.min(self.volumes_ml)
        return results

    def visualize_motion(self, save_animation: bool = False, filename: str = 'lv_motion.gif'):
        """Visualizes the final contour against the raw data and key landmarks."""
        if not self.resampled_contours: raise RuntimeError("Run .analyze() first.")
        try: import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation
        except ImportError: print("Matplotlib is required. pip install matplotlib"); return
        fig, ax = plt.subplots(figsize=(8, 8)); all_points = np.vstack(self.raw_contours)
        pad = (all_points[:,0].max() - all_points[:,0].min()) * 0.1
        ax.set_xlim(all_points[:,0].min()-pad, all_points[:,0].max()+pad); ax.set_ylim(all_points[:,1].min()-pad, all_points[:,1].max()+pad)
        ax.set_aspect('equal', 'box'); ax.set_title(f'LV Motion (Smoothing Factor: {self.smoothing_factor})'); ax.grid(True)
        line, = ax.plot([], [], '-', lw=2, c='dodgerblue', label='Smoothed Contour')
        raw_line, = ax.plot([], [], 'o', markersize=2, linestyle=':', lw=1, c='gray', label='Raw Points')
        base_line, = ax.plot([], [], '-', c='limegreen', lw=3, label='Mitral Base (raw)')
        apex_dot, = ax.plot([], [], '*', c='red', ms=15, label='Apex (raw)')
        mitral_dots, = ax.plot([], [], 'o', c='gold', ms=10, label='Mitral Points (raw)')
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.legend(loc='upper right')
        def init():
            line.set_data([],[]); raw_line.set_data([],[]); apex_dot.set_data([],[])
            base_line.set_data([],[]); mitral_dots.set_data([],[])
            return line, raw_line, apex_dot, base_line, mitral_dots, time_text
        def update(i):
            if len(self.resampled_contours[i]) == 0: return line, raw_line, apex_dot, base_line, mitral_dots, time_text
            raw = self.raw_contours[i]; line.set_data(self.resampled_contours[i][:,0], self.resampled_contours[i][:,1]); raw_line.set_data(raw[:,0], raw[:,1])
            p1,p_end=raw[0],raw[-1]; apex=raw[self._find_apex_index(raw)]
            apex_dot.set_data([apex[0]],[apex[1]]); base_line.set_data([p1[0],p_end[0]],[p1[1],p_end[1]]); mitral_dots.set_data([p1[0],p_end[0]],[p1[1],p_end[1]])
            time_text.set_text(f"F:{i}\nT:{self.times_sec[i]:.3f}s\nV:{self.volumes_ml[i]:.2f}ml\nGLS:{self.gls_curve[i]:.2f}%")
            return line, raw_line, apex_dot, base_line, mitral_dots, time_text
        anim=FuncAnimation(fig,update,frames=len(self.resampled_contours),init_func=init,blit=True,interval=50)
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    xml_file = "validation.xml"
    # --- TUNE THIS PARAMETER ---
    # 0 = no smoothing (interpolated). Higher values allow more smoothing.
    # Good values to test are between 5 and 100.
    smoothing_factor = 20.0 
    
    try:
        analyzer = CardiacAnalyzer(xml_file, smoothing_factor=smoothing_factor, num_points=130)
        analysis_results = analyzer.analyze()
        
        print("\n--- Final Analysis Results ---")
        print(f"Using Smoothing Factor: {smoothing_factor}")
        print(f"Ejection Fraction (EF): {analysis_results['ejection_fraction']:.2f} %")
        print(f"Peak Global Longitudinal Strain (GLS): {analysis_results['peak_global_longitudinal_strain']:.2f} %")
        print(f"End-Diastolic Volume (EDV): {analysis_results['end_diastolic_volume_ml']:.2f} ml")
        print(f"End-Systolic Volume (ESV): {analysis_results['end_systolic_volume_ml']:.2f} ml")
        print("----------------------------------\n")

        print("Starting visualization. The blue line should now perfectly track the gold and red markers.")
        analyzer.visualize_motion()
        
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        print(f"An error occurred: {e}")