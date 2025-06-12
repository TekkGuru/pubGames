import xml.etree.ElementTree as ET
from typing import List, Dict, Any

import numpy as np
from scipy.interpolate import splprep, splev

class CardiacAnalyzer:
    """
    Analyzes Left Ventricular (LV) function from contour data in an XML file.

    Calculates LV volume, Ejection Fraction (EF), and Global Longitudinal Strain (GLS)
    from time-resolved 2D endocardial contour points. This version uses an open-contour
    model with a flat mitral base, which is clinically more accurate.
    """

    def __init__(self, xml_file_path: str):
        """Initializes the analyzer by parsing the XML file."""
        self.xml_path = xml_file_path
        self._parse_xml_data()
        self.resampled_contours = []
        self.volumes_ml = np.array([])
        self.gls_curve = np.array([])
        self.L0 = 0.0

    def _parse_xml_data(self):
        """
        Parses the SpreadsheetML XML file to extract contour data and metadata.
        This function remains the same as it correctly extracts the raw data.
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
            first_cell_data = cells[0].find('ss:Data', ns)
            if first_cell_data is not None and first_cell_data.text:
                header = first_cell_data.text.strip()
                if header == 'frametime':
                    time_intervals_ms = [float(c.find('ss:Data', ns).text) for c in cells[1:] if c.find('ss:Data', ns) is not None]
                elif header == 'pixelsize':
                    self.pixel_size_mm = float(cells[1].find('ss:Data', ns).text)
                elif header in ['EndoX', 'EndoY', 'EpiX']:
                    parsing_state = header
                    continue

            if parsing_state == 'EndoX':
                row_data = [c.find('ss:Data', ns) for c in cells]
                if all(c is not None and c.get(f'{{{ns["ss"]}}}Type') == 'Number' for c in row_data):
                     x_coords_by_point.append([float(c.text) for c in row_data])
            elif parsing_state == 'EndoY':
                row_data = [c.find('ss:Data', ns) for c in cells]
                if all(c is not None and c.get(f'{{{ns["ss"]}}}Type') == 'Number' for c in row_data):
                    y_coords_by_point.append([float(c.text) for c in row_data])

        if not time_intervals_ms or not x_coords_by_point or not y_coords_by_point:
            raise ValueError("Could not find 'frametime', 'EndoX', or 'EndoY' data in the XML.")

        x_coords_by_frame = np.array(x_coords_by_point).T
        y_coords_by_frame = np.array(y_coords_by_point).T
        self.raw_contours = [np.column_stack((x, y)) for x, y in zip(x_coords_by_frame, y_coords_by_frame)]
        self.times_sec = np.cumsum([0] + time_intervals_ms) / 1000.0

    def _resample_contours(self, num_points: int = 165):
        """
        Resamples each raw contour using an OPEN cubic spline.
        This is a key correction: per=False treats the 49 points as an open
        myocardial wall, preserving the mitral valve endpoints.
        """
        self.resampled_contours = []
        for contour in self.raw_contours:
            # **CORRECTION**: Use per=False for an open spline. This prevents
            # the spline from curving between the first and last points.
            tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=False)
            
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck, der=0)
            
            self.resampled_contours.append(np.column_stack((x_new, y_new)))

    def _calculate_volumes(self):
        """
        **REWRITTEN**: Calculates LV volume using Modified Simpson's Rule (Method of Disks).
        
        This corrected method uses a flat plane at the mitral base as the cap
        of the volume, preventing massive overestimation and reflecting the
        true chamber geometry.
        """
        volumes_px3 = []
        for i, contour in enumerate(self.resampled_contours):
            # Landmarks are identified from the more sparse but original raw contour
            raw_contour = self.raw_contours[i]
            
            # 1. Define Mitral Base and find Apex
            mitral_p1, mitral_p49 = raw_contour[0], raw_contour[-1]
            mitral_base_line = mitral_p49 - mitral_p1
            
            # Distance from each point to the mitral base line
            norm_base_sq = np.dot(mitral_base_line, mitral_base_line)
            if norm_base_sq == 0:
                distances = np.linalg.norm(raw_contour - mitral_p1, axis=1)
            else:
                vecs_to_points = raw_contour - mitral_p1
                t = np.dot(vecs_to_points, mitral_base_line) / norm_base_sq
                projection = mitral_p1 + t[:, np.newaxis] * mitral_base_line
                distances = np.linalg.norm(vecs_to_points - projection, axis=1)
                
            apex = raw_contour[np.argmax(distances)]
            mitral_midpoint = (mitral_p1 + mitral_p49) / 2.0
            
            # 2. Define Long Axis (from apex to mitral midpoint)
            long_axis_vector = mitral_midpoint - apex
            long_axis_length = np.linalg.norm(long_axis_vector)
            if long_axis_length == 0:
                volumes_px3.append(0)
                continue
            
            # 3. Orient the chamber so the long axis is on the Y-axis and apex is at top
            angle = np.arctan2(long_axis_vector[0], long_axis_vector[1])
            rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], 
                                        [-np.sin(angle), np.cos(angle)]])
            
            # Use the high-resolution resampled contour for volume calculation
            # Translate so apex is at the origin, then rotate
            oriented_contour = np.dot(rotation_matrix, (contour - apex).T).T

            # 4. Integrate using Method of Disks
            total_volume = 0.0
            
            # The y-coordinates now represent the position along the long axis
            # Slices should only be between apex (y=0) and the base (y=long_axis_length)
            valid_points = oriented_contour[oriented_contour[:, 1] < long_axis_length]
            
            # Sort points by their position on the long axis (y-coordinate)
            sorted_indices = np.argsort(valid_points[:, 1])
            sorted_points = valid_points[sorted_indices]

            if len(sorted_points) < 2:
                volumes_px3.append(0)
                continue

            # Sum the volume of the small cylindrical disks (frustums)
            for j in range(len(sorted_points) - 1):
                # Radius is the absolute x-coordinate
                r1, r2 = abs(sorted_points[j, 0]), abs(sorted_points[j+1, 0])
                y1, y2 = sorted_points[j, 1], sorted_points[j+1, 1]
                
                disk_height = y2 - y1
                # Use average radius for the slice
                avg_radius = (r1 + r2) / 2.0
                
                # Formula for volume of a disk
                disk_volume = np.pi * (avg_radius**2) * disk_height
                if disk_volume > 0:
                    total_volume += disk_volume

            volumes_px3.append(total_volume)
        
        volumes_px3 = np.array(volumes_px3)
        self.volumes_ml = volumes_px3 * (self.pixel_size_mm**3) / 1000.0

    def _calculate_ef(self) -> float:
        """Calculates Ejection Fraction from the volume curve."""
        if len(self.volumes_ml) == 0:
            raise RuntimeError("Volumes must be calculated before EF.")
        edv = np.max(self.volumes_ml)
        esv = np.min(self.volumes_ml)
        return ((edv - esv) / edv) * 100.0 if edv > 0 else 0.0

    def _calculate_gls(self):
        """
        **CORRECTED**: Calculates GLS based on the length of the OPEN myocardial wall.
        """
        if not self.resampled_contours:
            raise RuntimeError("Contours must be resampled before GLS calculation.")
            
        def contour_length(contour: np.ndarray) -> float:
            contour_mm = contour * self.pixel_size_mm
            # Sum of distances between adjacent points along the open curve
            lengths = np.sqrt(np.sum(np.diff(contour_mm, axis=0)**2, axis=1))
            return np.sum(lengths)
        
        self.L0 = contour_length(self.resampled_contours[0])
        if self.L0 == 0:
            self.gls_curve = np.zeros(len(self.resampled_contours))
            return

        gls_values = [((contour_length(c) - self.L0) / self.L0) * 100.0 for c in self.resampled_contours]
        self.gls_curve = np.array(gls_values)

    def analyze(self) -> Dict[str, Any]:
        """Runs the full analysis pipeline and returns the results."""
        print("Step 1: Resampling contours (using open curve model)...")
        self._resample_contours()
        
        print("Step 2: Calculating volume curve (using Modified Simpson's method)...")
        self._calculate_volumes()
        
        print("Step 3: Calculating Ejection Fraction...")
        ef = self._calculate_ef()
        
        print("Step 4: Calculating Global Longitudinal Strain curve...")
        self._calculate_gls()
        
        peak_gls = np.min(self.gls_curve) if len(self.gls_curve) > 0 else 0.0
        
        print("\nAnalysis Complete.")
        
        return {
            'ejection_fraction': ef,
            'peak_global_longitudinal_strain': peak_gls,
            'end_diastolic_volume_ml': np.max(self.volumes_ml) if len(self.volumes_ml) > 0 else 0.0,
            'end_systolic_volume_ml': np.min(self.volumes_ml) if len(self.volumes_ml) > 0 else 0.0,
            'time_seconds': self.times_sec,
            'volume_ml_curve': self.volumes_ml,
            'gls_percent_curve': self.gls_curve,
        }
        
    def visualize_motion(self, use_resampled: bool = True, save_animation: bool = False, filename: str = 'lv_motion.gif'):
        """Creates an animation of the LV contour motion over the cardiac cycle."""
        # This visualization function from the previous step is still valid and useful.
        # It's included here for completeness.
        if not self.raw_contours or (use_resampled and not self.resampled_contours):
            raise RuntimeError("Please run the .analyze() method before visualizing motion.")
            
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Matplotlib is required for visualization. Please install it: pip install matplotlib")
            return

        contours_to_show = self.resampled_contours if use_resampled else self.raw_contours
        
        fig, ax = plt.subplots(figsize=(8, 8))
        all_points = np.vstack(contours_to_show)
        x_min, y_min, x_max, y_max = all_points[:,0].min(), all_points[:,1].min(), all_points[:,0].max(), all_points[:,1].max()
        padding = (x_max - x_min) * 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect('equal', 'box')
        ax.set_title('LV Endocardial Contour Motion')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True)

        line, = ax.plot([], [], '-', lw=2, color='dodgerblue') # Changed to '-' for open contour
        apex_dot, = ax.plot([], [], '*', color='red', markersize=15, label='Apex')
        base_line, = ax.plot([], [], '-', color='limegreen', lw=3, label='Mitral Base')
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.legend(loc='upper right')

        def init():
            line.set_data([], [])
            apex_dot.set_data([], [])
            base_line.set_data([], [])
            time_text.set_text('')
            return line, apex_dot, base_line, time_text

        def update(i):
            contour = contours_to_show[i]
            line.set_data(contour[:, 0], contour[:, 1])

            raw_contour = self.raw_contours[i]
            mitral_p1, mitral_p49 = raw_contour[0], raw_contour[-1]
            mitral_base_line = mitral_p49 - mitral_p1
            
            norm_base_sq = np.dot(mitral_base_line, mitral_base_line)
            if norm_base_sq == 0: distances = np.linalg.norm(raw_contour - mitral_p1, axis=1)
            else:
                vecs_to_points = raw_contour - mitral_p1
                t = np.dot(vecs_to_points, mitral_base_line) / norm_base_sq
                projection = mitral_p1 + t[:, np.newaxis] * mitral_base_line
                distances = np.linalg.norm(vecs_to_points - projection, axis=1)

            apex = raw_contour[np.argmax(distances)]

            apex_dot.set_data([apex[0]], [apex[1]])
            base_line.set_data([mitral_p1[0], mitral_p49[0]], [mitral_p1[1], mitral_p49[1]])
            
            frame_info = f"Frame: {i}\nTime: {self.times_sec[i]:.3f} s\nVolume: {self.volumes_ml[i]:.2f} ml\nGLS: {self.gls_curve[i]:.2f} %"
            time_text.set_text(frame_info)
            
            return line, apex_dot, base_line, time_text

        anim = FuncAnimation(fig, update, frames=len(contours_to_show), init_func=init, blit=True, interval=50)

        if save_animation:
            print(f"Saving animation to '{filename}'... This may take a moment.")
            anim.save(filename, writer='pillow', fps=20)
            print("Animation saved successfully.")
        else:
            plt.show()

# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    xml_file = "validation.xml"
    
    try:
        analyzer = CardiacAnalyzer(xml_file)
        analysis_results = analyzer.analyze()
        
        print("\n--- Corrected Results ---")
        print(f"Ejection Fraction (EF): {analysis_results['ejection_fraction']:.2f} %")
        print(f"Peak Global Longitudinal Strain (GLS): {analysis_results['peak_global_longitudinal_strain']:.2f} %")
        print(f"End-Diastolic Volume (EDV): {analysis_results['end_diastolic_volume_ml']:.2f} ml")
        print(f"End-Systolic Volume (ESV): {analysis_results['end_systolic_volume_ml']:.2f} ml")
        print("------------------------\n")

        print("Starting visualization of the cardiac cycle motion.")
        analyzer.visualize_motion(use_resampled=True)
        
    except FileNotFoundError:
        print(f"Error: The file '{xml_file}' was not found.")
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"An error occurred: {e}")