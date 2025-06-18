import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.interpolate import splprep, splev

class CardiacAnalyzer:
    """
    Corrected version of the LV analysis script. This version implements a fixed-apex
    strategy and includes a more robust XML parser and a corrected volume calculation.
    """

    def __init__(self, xml_file_path: str, smoothing_factor: float = 20.0, num_points: int = 49):
        self.xml_path = xml_file_path
        self.smoothing_factor = smoothing_factor
        self.num_points = num_points
        self._parse_xml_data()
        self.resampled_contours = []
        self.volumes_ml = np.array([])
        self.gls_curve = np.array([])
        # This will store the apex index from the reference frame (frame 0).
        self.apex_index_ref = None
        # Attributes for visualization of segment lengths
        self.total_lengths_mm = None
        self.all_segment_lengths_mm = None
        self.min_len_frame_idx = None
        self.max_len_frame_idx = None


    def _parse_xml_data(self):
        """
        Parses the SpreadsheetML XML file.
        **FIX**: Added robustness to handle potentially ragged data arrays coming
        from Excel by determining a consistent number of frames.
        """
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {e}")
        all_rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', ns)
        time_intervals_ms, x_coords_by_point, y_coords_by_point = [], [], []
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
                elif header in ['EndoX', 'EndoY', 'EpiX']:
                    parsing_state = header
                    continue
            
            if parsing_state in ['EndoX', 'EndoY']:
                numeric_cells = [c.find('ss:Data', ns) for c in cells if c.find('ss:Data', ns) is not None and c.find('ss:Data', ns).get(f'{{{ns["ss"]}}}Type') == 'Number']
                if not numeric_cells: continue
                
                row_values = [float(c.text) for c in numeric_cells]
                
                if parsing_state == 'EndoX':
                    x_coords_by_point.append(row_values)
                elif parsing_state == 'EndoY':
                    y_coords_by_point.append(row_values)

        if not time_intervals_ms or not x_coords_by_point or not y_coords_by_point:
            raise ValueError("Could not find required data (frametime, EndoX, EndoY) in XML.")
        
        num_frames = min(len(row) for row in x_coords_by_point)
        print(f"Detected a consistent number of {num_frames} frames.")

        x_coords_trimmed = [row[:num_frames] for row in x_coords_by_point]
        y_coords_trimmed = [row[:num_frames] for row in y_coords_by_point]

        x_coords_by_frame = np.array(x_coords_trimmed).T
        y_coords_by_frame = np.array(y_coords_trimmed).T

        self.raw_contours = [np.column_stack((x, y)) for x, y in zip(x_coords_by_frame, y_coords_by_frame)]
        self.times_sec = np.cumsum([0] + time_intervals_ms[:num_frames-1]) / 1000.0

    def _find_apex_index(self, contour: np.ndarray) -> int:
        """
        Determines the apex as the point on the contour farthest from the
        center of the two mitral valve endpoints.
        """
        if len(contour) < 2:
            return 0
        mitral_p1, mitral_p_end = contour[0], contour[-1]
        base_center = (mitral_p1 + mitral_p_end) / 2.0
        distances = np.linalg.norm(contour - base_center, axis=1)
        return np.argmax(distances)

    def _determine_and_validate_apex_on_ref_frame(self):
        """
        Determines and validates the apex on the reference frame (frame 0).
        """
        if not self.raw_contours:
            raise ValueError("Raw contours not loaded. Cannot determine apex.")
        
        ref_contour = self.raw_contours[0]
        apex_idx = self._find_apex_index(ref_contour)

        if apex_idx == 0 or apex_idx == len(ref_contour) - 1:
            raise ValueError(
                "Apex detected as an endpoint on the reference frame. "
                "Cannot perform stable two-segment analysis. Please check input data."
            )
        self.apex_index_ref = apex_idx
        print(f"Reference apex index set to: {self.apex_index_ref}")

    def _resample_contours(self):
        """
        Implements a fixed-apex "Anchor-and-Smooth" strategy.
        """
        self.resampled_contours = []

        def get_spline_and_length(segment_pts, smoothing_factor):
            n_pts = len(segment_pts)
            if n_pts < 2: return None, 0.0
            k = min(3, n_pts - 1)
            s = smoothing_factor if k >= 3 else 0
            try:
                tck, _ = splprep([segment_pts[:, 0], segment_pts[:, 1]], s=s, k=k, per=False)
                u_fine = np.linspace(0, 1, 200)
                points_fine = np.column_stack(splev(u_fine, tck))
                length = np.sum(np.linalg.norm(np.diff(points_fine, axis=0), axis=1))
                return tck, length
            except Exception as e:
                print(f"Warning: Spline generation failed for a segment. {e}")
                return None, 0.0

        for raw_contour in self.raw_contours:
            apex_idx = self.apex_index_ref
            seg1_pts = raw_contour[:apex_idx + 1]
            seg2_pts = raw_contour[apex_idx:]

            tck1, len1 = get_spline_and_length(seg1_pts, self.smoothing_factor)
            tck2, len2 = get_spline_and_length(seg2_pts, self.smoothing_factor)

            if tck1 is None or tck2 is None:
                self.resampled_contours.append(np.array([])); continue
            
            total_len = len1 + len2
            if total_len <= 1e-9:
                self.resampled_contours.append(np.array([])); continue
                
            num_spline_points = self.num_points + 1
            n_pts1 = int(round(num_spline_points * len1 / total_len))
            n_pts2 = num_spline_points - n_pts1

            if n_pts1 < 2: n_pts1 = 2; n_pts2 = num_spline_points - 2
            if n_pts2 < 2: n_pts2 = 2; n_pts1 = num_spline_points - 2
            if n_pts1 < 2 or n_pts2 < 2:
                self.resampled_contours.append(np.array([])); continue

            u1_new = np.linspace(0, 1, n_pts1)
            u2_new = np.linspace(0, 1, n_pts2)
            seg1_new = np.column_stack(splev(u1_new, tck1))
            seg2_new = np.column_stack(splev(u2_new, tck2))
            
            final_contour = np.vstack([seg1_new[:-1], seg2_new])
            self.resampled_contours.append(final_contour)

    def _calculate_all_lengths(self):
        """
        Calculates total and segmental lengths for all contours and finds the
        min and max length frames for visualization.
        """
        if not self.resampled_contours:
            raise RuntimeError("Resampling must be done before calculating lengths.")

        self.total_lengths_mm = []
        self.all_segment_lengths_mm = []

        for contour in self.resampled_contours:
            if contour.shape[0] < 2:
                self.total_lengths_mm.append(0)
                self.all_segment_lengths_mm.append(np.array([]))
                continue

            contour_mm = contour * self.pixel_size_mm
            segment_lengths = np.linalg.norm(np.diff(contour_mm, axis=0), axis=1)
            self.all_segment_lengths_mm.append(segment_lengths)
            self.total_lengths_mm.append(np.sum(segment_lengths))

        self.total_lengths_mm = np.array(self.total_lengths_mm)
        if len(self.total_lengths_mm) > 0:
            self.min_len_frame_idx = np.argmin(self.total_lengths_mm)
            self.max_len_frame_idx = 0  # Assuming frame 0 is End-Diastole (max length)
            print(f"Max length frame (ED): {self.max_len_frame_idx}, Min length frame (ES): {self.min_len_frame_idx}")


    def _calculate_volumes(self):
        """
        Calculates volume using method of disks, PARALLEL to the mitral base.
        """
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

    def _calculate_gls(self):
        """
        Calculates Global Longitudinal Strain (GLS) using pre-calculated total lengths.
        """
        if self.total_lengths_mm is None or len(self.total_lengths_mm) == 0:
            self.gls_curve = np.zeros(len(self.resampled_contours))
            return

        ref_total_length = self.total_lengths_mm[self.max_len_frame_idx]

        if ref_total_length < 1e-9:
            self.gls_curve = np.zeros(len(self.resampled_contours))
            return
            
        gls_values = []
        for curr_total_length in self.total_lengths_mm:
            strain = (curr_total_length - ref_total_length) / ref_total_length
            gls_values.append(strain * 100.0)
            
        self.gls_curve = np.array(gls_values)

    def _calculate_ef(self):
        """Calculates Ejection Fraction from the volume curve."""
        if len(self.volumes_ml) == 0 or np.max(self.volumes_ml) == 0: return 0.0
        edv = np.max(self.volumes_ml)
        esv = np.min(self.volumes_ml)
        return ((edv - esv) / edv) * 100.0
        
    def analyze(self) -> Dict[str, Any]:
        """Runs the full, corrected analysis pipeline."""
        print("Step 1: Determining and validating apex on reference frame...")
        self._determine_and_validate_apex_on_ref_frame()
        print(f"Step 2: Resampling all contours with fixed-apex method (s={self.smoothing_factor})...")
        self._resample_contours()
        print("Step 3: Calculating all contour lengths for GLS and visualization...")
        self._calculate_all_lengths()
        print("Step 4: Calculating volume curve with corrected orientation...")
        self._calculate_volumes()
        print("Step 5: Calculating Ejection Fraction...")
        ef = self._calculate_ef()
        print("Step 6: Calculating Global Longitudinal Strain from total contour length...")
        self._calculate_gls()
        print("\nAnalysis Complete.")
        
        results = {
            'ejection_fraction': ef,
            'peak_global_longitudinal_strain': 0.0,
            'end_diastolic_volume_ml': 0.0,
            'end_systolic_volume_ml': 0.0,
            'time_seconds': self.times_sec,
            'volume_ml_curve': self.volumes_ml,
            'gls_percent_curve': self.gls_curve
        }
        if len(self.gls_curve) > 0:
            results['peak_global_longitudinal_strain'] = np.min(self.gls_curve)
        if len(self.volumes_ml) > 0:
            results['end_diastolic_volume_ml'] = np.max(self.volumes_ml)
            results['end_systolic_volume_ml'] = np.min(self.volumes_ml)
        return results

    def visualize_motion(self, save_animation: bool = False, filename: str = 'lv_motion.gif'):
        """Visualizes the final contour against the raw data and key landmarks."""
        if not self.resampled_contours:
            raise RuntimeError("Run .analyze() first.")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib.widgets import Button, Slider
        except ImportError:
            print("Matplotlib is required for visualization. Please run: pip install matplotlib")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)  # Make room for widgets

        all_points = np.vstack(self.raw_contours)
        pad = (all_points[:,0].max() - all_points[:,0].min()) * 0.1
        ax.set_xlim(all_points[:,0].min()-pad, all_points[:,0].max()+pad)
        ax.set_ylim(all_points[:,1].min()-pad, all_points[:,1].max()+pad)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'LV Motion (Smoothing Factor: {self.smoothing_factor})')
        ax.grid(True)

        line, = ax.plot([], [], '-', lw=2, c='dodgerblue', label='Smoothed Contour')
        raw_line, = ax.plot([], [], 'o', markersize=2, linestyle=':', lw=1, c='gray', label='Raw Points')
        base_line, = ax.plot([], [], '-', c='limegreen', lw=3, label='Mitral Base (raw)')
        ref_apex_pt = self.raw_contours[0][self.apex_index_ref]
        apex_dot, = ax.plot([ref_apex_pt[0]], [ref_apex_pt[1]], '*', c='red', ms=15, label=f'Apex (Ref Frame {self.apex_index_ref})')
        mitral_dots, = ax.plot([], [], 'o', c='gold', ms=10, label='Mitral Points (raw)')
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.legend(loc='upper right')

        self.segment_labels = []

        # --- Animation Control ---
        is_paused = [False] # Use a list to make it mutable inside callbacks

        def update(frame_index):
            i = int(frame_index) # Ensure frame_index is an integer
            for label in self.segment_labels:
                label.remove()
            self.segment_labels.clear()

            if len(self.resampled_contours[i]) == 0:
                return
            
            raw = self.raw_contours[i]
            line.set_data(self.resampled_contours[i][:,0], self.resampled_contours[i][:,1])
            raw_line.set_data(raw[:,0], raw[:,1])
            p1, p_end = raw[0], raw[-1]
            base_line.set_data([p1[0], p_end[0]], [p1[1], p_end[1]])
            mitral_dots.set_data([p1[0], p_end[0]], [p1[1], p_end[1]])
            time_text.set_text(f"Frame: {i}\nTime: {self.times_sec[i]:.3f}s\nVolume: {self.volumes_ml[i]:.2f}ml\nGLS: {self.gls_curve[i]:.2f}%")

            #if i == self.min_len_frame_idx or i == self.max_len_frame_idx:
            min_lengths = self.all_segment_lengths_mm[self.min_len_frame_idx]
            max_lengths = self.all_segment_lengths_mm[self.max_len_frame_idx]
            
            with open('valueDumpMAX.txt', 'w') as file:
                for item in max_lengths:
                    file.write(f"{item}\n")
                    
            with open('valueDumpMIN.txt', 'w') as file:
                for item in min_lengths:
                    file.write(f"{item}\n")
                    
            if len(min_lengths) == len(max_lengths):
                current_contour = self.resampled_contours[i]
                for j in range(len(current_contour) - 1):
                    p1, p2 = current_contour[j], current_contour[j+1]
                    midpoint = (p1 + p2) / 2.0
                    label_text = f"{self.all_segment_lengths_mm[i][j]:.4g}"
                    text_artist = ax.text(midpoint[0], midpoint[1], label_text, fontsize=6, color='white', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6), zorder=10)
                    self.segment_labels.append(text_artist)
            fig.canvas.draw_idle()

        anim = FuncAnimation(fig, update, frames=len(self.resampled_contours), interval=50)

        # --- Widgets ---
        ax_pause = plt.axes([0.7, 0.025, 0.1, 0.04])
        btn_pause = Button(ax_pause, 'Pause', color='lightgoldenrodyellow', hovercolor='0.975')

        def toggle_pause(event):
            if is_paused[0]:
                anim.resume()
                btn_pause.label.set_text('Pause')
            else:
                anim.pause()
                btn_pause.label.set_text('Play')
            is_paused[0] = not is_paused[0]
            fig.canvas.draw_idle()

        btn_pause.on_clicked(toggle_pause)

        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        frame_slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=len(self.resampled_contours) - 1,
            valinit=0,
            valstep=1
        )
        
        def set_frame(val):
            if not is_paused[0]:
                toggle_pause(None) # Pause the animation if slider is used
            update(val)

        frame_slider.on_changed(set_frame)

        update(0) # Initialize the plot to the first frame
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    xml_file = "validation.xml" 
    
    smoothing_factor = 20.0 
    
    try:
        analyzer = CardiacAnalyzer(xml_file, smoothing_factor=smoothing_factor, num_points=49)
        analysis_results = analyzer.analyze()
        
        print("\n--- Final Analysis Results ---")
        print(f"Using Smoothing Factor: {smoothing_factor}")
        print(f"Ejection Fraction (EF): {analysis_results['ejection_fraction']:.2f} %")
        print(f"Peak Global Longitudinal Strain (GLS): {analysis_results['peak_global_longitudinal_strain']:.2f} %")
        print(f"End-Diastolic Volume (EDV): {analysis_results['end_diastolic_volume_ml']:.2f} ml")
        print(f"End-Systolic Volume (ESV): {analysis_results['end_systolic_volume_ml']:.2f} ml")
        print("----------------------------------\n")

        print("Starting visualization...")
        analyzer.visualize_motion()
        
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        print(f"\nAN ERROR OCCURRED: {e}")
