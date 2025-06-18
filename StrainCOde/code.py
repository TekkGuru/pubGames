import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.interpolate import splprep, splev

class CardiacAnalyzer:
    """
    Advanced version of the LV analysis script. This version includes endocardial and
    epicardial GLS, strain rate, segmental strain analysis, and bulls-eye plotting.
    """

    def __init__(self, xml_file_path: str, smoothing_factor: float = 20.0, num_points: int = 130):
        self.xml_path = xml_file_path
        self.smoothing_factor = smoothing_factor
        self.num_points = num_points
        self._parse_xml_data()
        
        # Endo data
        self.resampled_contours = []
        self.volumes_ml = np.array([])
        self.gls_curve = np.array([])
        self.strain_rate_curve = np.array([])
        self.segmental_strain_results = {}

        # Epi data
        self.resampled_epi_contours = []
        self.epi_gls_curve = np.array([])

        self.apex_index_ref = None
        self.min_len_frame_idx = None
        self.max_len_frame_idx = 0

    def _parse_xml_data(self):
        """
        Parses the SpreadsheetML XML file for both Endo and Epi contours.
        """
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {e}")
        all_rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', ns)
        
        time_intervals_ms, endo_x, endo_y, epi_x, epi_y = [], [], [], [], []
        parsing_state = None

        for row in all_rows:
            cells = row.findall('ss:Cell', ns)
            if not cells: continue
            first_cell_data_element = cells[0].find('ss:Data', ns)
            
            if first_cell_data_element is not None and first_cell_data_element.text:
                header = first_cell_data_element.text.strip()
                if header == 'frametime':
                    time_intervals_ms = [float(c.find('ss:Data', ns).text) for c in cells[1:] if c.find('ss:Data', ns) is not None]
                elif header == 'pixelsize':
                    self.pixel_size_mm = float(cells[1].find('ss:Data', ns).text)
                elif header in ['EndoX', 'EndoY', 'EpiX', 'EpiY']:
                    parsing_state = header
                    continue
            
            if parsing_state:
                numeric_cells = [c.find('ss:Data', ns) for c in cells if c.find('ss:Data', ns) is not None and c.find('ss:Data', ns).get(f'{{{ns["ss"]}}}Type') == 'Number']
                if not numeric_cells: continue
                row_values = [float(c.text) for c in numeric_cells]
                
                if parsing_state == 'EndoX': endo_x.append(row_values)
                elif parsing_state == 'EndoY': endo_y.append(row_values)
                elif parsing_state == 'EpiX': epi_x.append(row_values)
                elif parsing_state == 'EpiY': epi_y.append(row_values)

        if not all([time_intervals_ms, endo_x, endo_y, epi_x, epi_y]):
            raise ValueError("Could not find required data (frametime, EndoX/Y, EpiX/Y) in XML.")

        # Process Endo and Epi contours
        self.raw_contours, self.times_sec = self._process_raw_contours(endo_x, endo_y, time_intervals_ms)
        self.raw_epi_contours, _ = self._process_raw_contours(epi_x, epi_y, time_intervals_ms)

    def _process_raw_contours(self, x_coords, y_coords, times):
        """Helper to process raw coordinate lists into contours."""
        num_frames = min(len(row) for row in x_coords)
        x_trimmed = [row[:num_frames] for row in x_coords]
        y_trimmed = [row[:num_frames] for row in y_coords]
        x_by_frame = np.array(x_trimmed).T
        y_by_frame = np.array(y_trimmed).T
        contours = [np.column_stack((x, y)) for x, y in zip(x_by_frame, y_by_frame)]
        times_sec = np.cumsum([0] + times[:num_frames-1]) / 1000.0
        return contours, times_sec

    def _find_apex_index(self, contour: np.ndarray) -> int:
        if len(contour) < 2: return 0
        mitral_p1, mitral_p_end = contour[0], contour[-1]
        base_center = (mitral_p1 + mitral_p_end) / 2.0
        distances = np.linalg.norm(contour - base_center, axis=1)
        return np.argmax(distances)

    def _determine_and_validate_apex_on_ref_frame(self):
        if not self.raw_contours: raise ValueError("Endo contours not loaded.")
        ref_contour = self.raw_contours[0]
        apex_idx = self._find_apex_index(ref_contour)
        if apex_idx == 0 or apex_idx == len(ref_contour) - 1:
            raise ValueError("Apex detected as an endpoint on the reference frame.")
        self.apex_index_ref = apex_idx
        print(f"Reference apex index set to: {self.apex_index_ref}")

    def _resample_all_contours(self):
        print("Resampling Endocardial contours...")
        self.resampled_contours, self.n_pts_wall1, self.n_pts_wall2 = self._perform_resampling(self.raw_contours)
        print("Resampling Epicardial contours...")
        self.resampled_epi_contours, _, _ = self._perform_resampling(self.raw_epi_contours)

    def _perform_resampling(self, raw_contours_list):
        resampled_list = []
        n_pts1_ref, n_pts2_ref = 0, 0
        
        for i, raw_contour in enumerate(raw_contours_list):
            apex_idx = self.apex_index_ref
            seg1_pts = raw_contour[:apex_idx + 1]
            seg2_pts = raw_contour[apex_idx:]

            tck1, len1 = self._get_spline_and_length(seg1_pts)
            tck2, len2 = self._get_spline_and_length(seg2_pts)

            if tck1 is None or tck2 is None:
                resampled_list.append(np.array([])); continue
            
            total_len = len1 + len2
            if total_len <= 1e-9:
                resampled_list.append(np.array([])); continue
                
            num_spline_points = self.num_points + 1
            n_pts1 = int(round(num_spline_points * len1 / total_len))
            n_pts2 = num_spline_points - n_pts1

            if n_pts1 < 2: n_pts1 = 2; n_pts2 = num_spline_points - 2
            if n_pts2 < 2: n_pts2 = 2; n_pts1 = num_spline_points - 2
            if n_pts1 < 2 or n_pts2 < 2:
                resampled_list.append(np.array([])); continue
            
            if i == 0: n_pts1_ref, n_pts2_ref = n_pts1, n_pts2

            u1_new = np.linspace(0, 1, n_pts1); u2_new = np.linspace(0, 1, n_pts2)
            seg1_new = np.column_stack(splev(u1_new, tck1))
            seg2_new = np.column_stack(splev(u2_new, tck2))
            
            final_contour = np.vstack([seg1_new[:-1], seg2_new])
            resampled_list.append(final_contour)
            
        return resampled_list, n_pts1_ref, n_pts2_ref

    def _get_spline_and_length(self, segment_pts):
        n_pts = len(segment_pts)
        if n_pts < 2: return None, 0.0
        k = min(3, n_pts - 1)
        s = self.smoothing_factor if k >= 3 else 0
        try:
            tck, _ = splprep([segment_pts[:, 0], segment_pts[:, 1]], s=s, k=k, per=False)
            u_fine = np.linspace(0, 1, 200)
            points_fine = np.column_stack(splev(u_fine, tck))
            length = np.sum(np.linalg.norm(np.diff(points_fine, axis=0), axis=1))
            return tck, length
        except Exception: return None, 0.0
    
    def _calculate_all_lengths(self, contours_list):
        total_lengths = []
        all_segment_lengths = []
        for contour in contours_list:
            if contour.shape[0] < 2:
                total_lengths.append(0); all_segment_lengths.append(np.array([]))
                continue
            contour_mm = contour * self.pixel_size_mm
            segment_lengths = np.linalg.norm(np.diff(contour_mm, axis=0), axis=1)
            all_segment_lengths.append(segment_lengths)
            total_lengths.append(np.sum(segment_lengths))
        return np.array(total_lengths), all_segment_lengths

    def _calculate_volumes(self):
        self.volumes_ml = np.zeros(len(self.resampled_contours))
        for i, contour in enumerate(self.resampled_contours):
            if contour.shape[0] < 3: continue
            raw_contour = self.raw_contours[i]
            mitral_p1, mitral_p_end = raw_contour[0], raw_contour[-1]
            base_vector = mitral_p_end - mitral_p1
            if np.linalg.norm(base_vector) == 0: continue
            angle = np.arctan2(base_vector[1], base_vector[0])
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = np.array([[c, s], [-s, c]])
            oriented_contour = np.dot(rot_mat, (contour - mitral_p1).T).T
            y_coords = oriented_contour[:, 1]
            if np.max(y_coords) - np.min(y_coords) <= 1e-6: continue
            total_volume = 0.0
            sorted_points = oriented_contour[np.argsort(y_coords)]
            for j in range(len(sorted_points) - 1):
                y1, y2 = sorted_points[j, 1], sorted_points[j+1, 1]
                disk_height = y2 - y1
                if disk_height <= 0: continue
                slice_points = oriented_contour[(oriented_contour[:, 1] >= y1) & (oriented_contour[:, 1] <= y2)]
                if len(slice_points) < 2: continue
                radius = (np.max(slice_points[:, 0]) - np.min(slice_points[:, 0])) / 2.0
                total_volume += np.pi * (radius**2) * disk_height
            self.volumes_ml[i] = total_volume * (self.pixel_size_mm**3) / 1000.0
        
        self.min_vol_frame_idx = np.argmin(self.volumes_ml) if len(self.volumes_ml) > 0 else 0

    def _calculate_gls(self):
        self.total_lengths_mm, self.all_segment_lengths_mm = self._calculate_all_lengths(self.resampled_contours)
        self.min_len_frame_idx = np.argmin(self.total_lengths_mm) if len(self.total_lengths_mm) > 0 else 0
        ref_length = self.total_lengths_mm[self.max_len_frame_idx]
        if ref_length > 1e-9:
            self.gls_curve = (self.total_lengths_mm - ref_length) / ref_length * 100.0

        epi_total_lengths, _ = self._calculate_all_lengths(self.resampled_epi_contours)
        ref_epi_length = epi_total_lengths[self.max_len_frame_idx]
        if ref_epi_length > 1e-9:
            self.epi_gls_curve = (epi_total_lengths - ref_epi_length) / ref_epi_length * 100.0

    def _calculate_strain_rate(self):
        if len(self.gls_curve) > 1 and len(self.times_sec) == len(self.gls_curve):
            self.strain_rate_curve = np.gradient(self.gls_curve, self.times_sec)
    
    def _calculate_segmental_strain(self):
        ref_lengths = self.all_segment_lengths_mm[self.max_len_frame_idx]
        systolic_lengths = self.all_segment_lengths_mm[self.min_len_frame_idx]

        if len(ref_lengths) != len(systolic_lengths): return
        
        strains = (systolic_lengths - ref_lengths) / ref_lengths * 100.0
        
        num_segs_wall1 = self.n_pts_wall1 - 1
        basal1_end = num_segs_wall1 // 3
        mid1_end = basal1_end * 2
        
        num_segs_wall2 = self.n_pts_wall2 - 1
        basal2_start = num_segs_wall1 + num_segs_wall2 - (num_segs_wall2 // 3)
        mid2_start = basal2_start - (num_segs_wall2 // 3)

        self.segmental_strain_results = {
            'Basal Septal': np.mean(strains[:basal1_end]),
            'Mid Septal': np.mean(strains[basal1_end:mid1_end]),
            'Apical Septal': np.mean(strains[mid1_end:num_segs_wall1-1]),
            'Apical Cap': strains[num_segs_wall1-1],
            'Apical Lateral': np.mean(strains[num_segs_wall1:mid2_start]),
            'Mid Lateral': np.mean(strains[mid2_start:basal2_start]),
            'Basal Lateral': np.mean(strains[basal2_start:])
        }

    def _calculate_ef(self):
        if len(self.volumes_ml) == 0 or np.max(self.volumes_ml) == 0: return 0.0
        edv = self.volumes_ml[self.max_len_frame_idx]
        esv = self.volumes_ml[self.min_vol_frame_idx]
        return ((edv - esv) / edv) * 100.0 if edv > 0 else 0.0
        
    def analyze(self) -> Dict[str, Any]:
        """Runs the full, corrected analysis pipeline."""
        print("Step 1: Determining apex...")
        self._determine_and_validate_apex_on_ref_frame()
        print("Step 2: Resampling contours...")
        self._resample_all_contours()
        print("Step 3: Calculating volumes...")
        self._calculate_volumes()
        print("Step 4: Calculating GLS (Endo & Epi)...")
        self._calculate_gls()
        print("Step 5: Calculating Strain Rate...")
        self._calculate_strain_rate()
        print("Step 6: Calculating Segmental Strain...")
        self._calculate_segmental_strain()
        print("Step 7: Calculating Ejection Fraction...")
        ef = self._calculate_ef()
        print("\nAnalysis Complete.")
        
        results = {
            'ejection_fraction': ef,
            'peak_global_longitudinal_strain': np.min(self.gls_curve) if len(self.gls_curve) > 0 else 0,
            'peak_epi_gls': np.min(self.epi_gls_curve) if len(self.epi_gls_curve) > 0 else 0,
            'peak_strain_rate': np.min(self.strain_rate_curve) if len(self.strain_rate_curve) > 0 else 0,
            'end_diastolic_volume_ml': self.volumes_ml[self.max_len_frame_idx] if len(self.volumes_ml) > 0 else 0,
            'end_systolic_volume_ml': self.volumes_ml[self.min_vol_frame_idx] if len(self.volumes_ml) > 0 else 0,
            'segmental_results': self.segmental_strain_results
        }
        return results

    def visualize_bulls_eye(self):
        """Generates a 17-segment bulls-eye plot for segmental strain."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("Matplotlib is required for visualization.")
            return

        aha_map = {
            'Basal Septal': 2, 'Mid Septal': 8, 'Apical Septal': 14,
            'Basal Lateral': 5, 'Mid Lateral': 11, 'Apical Lateral': 16,
            'Apical Cap': 17
        }

        strain_data = {aha_map[key]: val for key, val in self.segmental_strain_results.items()}
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_yticklabels([]); ax.set_xticklabels([]); ax.spines['polar'].set_visible(False)

        cmap = plt.get_cmap('RdYlGn_r'); norm = mcolors.Normalize(vmin=-30, vmax=0)

        radii = [0.25, 0.65, 1.0]; angles = np.deg2rad(np.array([210, 270, 330, 30, 90, 150, 210]))

        for i in range(1, 18):
            color = cmap(norm(strain_data.get(i, -15)));
            if i not in strain_data: color = 'lightgrey'
                
            if i <= 6: ax.bar(angles[i-1], radii[2]-radii[1], width=np.deg2rad(60), bottom=radii[1], color=color, edgecolor='white', lw=2)
            elif i <= 12: ax.bar(angles[i-7], radii[1]-radii[0], width=np.deg2rad(60), bottom=radii[0], color=color, edgecolor='white', lw=2)
            elif i <= 16: ax.bar(angles[i-13], radii[0], width=np.deg2rad(60), color=color, edgecolor='white', lw=2)
            else: ax.bar(0, radii[0], width=np.deg2rad(360), color=color, edgecolor='white', lw=1)
                
            if i in strain_data:
                angle_mid = angles[i-1] + np.deg2rad(30) if i <= 6 else angles[i-7] + np.deg2rad(30) if i <= 12 else angles[i-13] + np.deg2rad(30) if i <=16 else 0
                radius_mid = (radii[2]+radii[1])/2 if i <=6 else (radii[1]+radii[0])/2 if i<=12 else radii[0]/2
                if i == 17: radius_mid = 0
                ax.text(angle_mid, radius_mid, f"{strain_data[i]:.1f}", ha='center', va='center', fontsize=10, color='black')

        ax.set_rmax(1.0); fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1, label='Peak Systolic Strain (%)')
        ax.set_title('AHA 17-Segment Model (A4C View)'); plt.show()

    def visualize_motion(self, save_animation: bool = False, filename: str = 'lv_motion.gif'):
        if not self.resampled_contours: raise RuntimeError("Run .analyze() first.")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib.widgets import Button, Slider
        except ImportError:
            print("Matplotlib is required for visualization.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8)); plt.subplots_adjust(bottom=0.2)
        all_points = np.vstack(self.raw_contours); pad = (all_points[:,0].max() - all_points[:,0].min()) * 0.1
        ax.set_xlim(all_points[:,0].min()-pad, all_points[:,0].max()+pad); ax.set_ylim(all_points[:,1].min()-pad, all_points[:,1].max()+pad)
        ax.set_aspect('equal', 'box'); ax.set_title('LV Motion'); ax.grid(True)

        line, = ax.plot([], [], '-', lw=2, c='dodgerblue', label='Endo Contour')
        epi_line, = ax.plot([], [], '-', lw=2, c='cyan', label='Epi Contour')
        raw_line, = ax.plot([], [], 'o', markersize=2, linestyle=':', lw=1, c='gray', label='Raw Points')
        base_line, = ax.plot([], [], '-', c='limegreen', lw=3, label='Mitral Base (raw)')
        ref_apex_pt = self.raw_contours[0][self.apex_index_ref]
        apex_dot, = ax.plot([ref_apex_pt[0]], [ref_apex_pt[1]], '*', c='red', ms=15, label=f'Apex')
        mitral_dots, = ax.plot([], [], 'o', c='gold', ms=10, label='Mitral Points (raw)')
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.legend(loc='upper right')

        is_paused = [False]

        def update(frame_index):
            i = int(frame_index)
            line.set_data(self.resampled_contours[i][:,0], self.resampled_contours[i][:,1])
            epi_line.set_data(self.resampled_epi_contours[i][:,0], self.resampled_epi_contours[i][:,1])
            raw = self.raw_contours[i]; raw_line.set_data(raw[:,0], raw[:,1])
            p1, p_end = raw[0], raw[-1]; base_line.set_data([p1[0], p_end[0]], [p1[1], p_end[1]])
            mitral_dots.set_data([p1[0], p_end[0]], [p1[1], p_end[1]])
            time_text.set_text(f"Frame: {i}\nTime: {self.times_sec[i]:.3f}s\nVolume: {self.volumes_ml[i]:.2f}ml\nGLS: {self.gls_curve[i]:.2f}%")
            fig.canvas.draw_idle()

        anim = FuncAnimation(fig, update, frames=len(self.resampled_contours), blit=False, interval=50)
        ax_pause = plt.axes([0.7, 0.05, 0.1, 0.04]); btn_pause = Button(ax_pause, 'Pause')
        def toggle_pause(event):
            if is_paused[0]: anim.resume(); btn_pause.label.set_text('Pause')
            else: anim.pause(); btn_pause.label.set_text('Play')
            is_paused[0] = not is_paused[0]
        btn_pause.on_clicked(toggle_pause)

        ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03]); frame_slider = Slider(ax_slider, 'Frame', 0, len(self.resampled_contours)-1, valinit=0, valstep=1)
        def set_frame(val):
            if not is_paused[0]: toggle_pause(None)
            update(val)
        frame_slider.on_changed(set_frame)

        update(0)
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    xml_file = "validation.xml" 
    smoothing_factor = 20.0 
    
    try:
        analyzer = CardiacAnalyzer(xml_file, smoothing_factor=smoothing_factor, num_points=130)
        analysis_results = analyzer.analyze()
        
        print("\n--- Final Analysis Results ---")
        print(f"  Ejection Fraction (EF): {analysis_results['ejection_fraction']:.2f} %")
        print(f"  Peak Endo GLS: {analysis_results['peak_global_longitudinal_strain']:.2f} %")
        print(f"  Peak Epi GLS: {analysis_results['peak_epi_gls']:.2f} %")
        print(f"  Peak Strain Rate: {analysis_results['peak_strain_rate']:.2f} %/s")
        print(f"  End-Diastolic Volume (EDV): {analysis_results['end_diastolic_volume_ml']:.2f} ml")
        print(f"  End-Systolic Volume (ESV): {analysis_results['end_systolic_volume_ml']:.2f} ml")
        print("\n--- Peak Segmental Strain (%) ---")
        for segment, strain in analysis_results['segmental_results'].items():
            print(f"  {segment:<15}: {strain:.2f}")
        print("----------------------------------\n")

        print("Starting interactive motion visualization...")
        analyzer.visualize_motion()
        
        print("Displaying segmental strain bulls-eye plot...")
        analyzer.visualize_bulls_eye()
        
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        print(f"\nAN ERROR OCCURRED: {e}")
