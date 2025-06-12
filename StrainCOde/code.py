import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Any

import numpy as np
from scipy.interpolate import splprep, splev

class CardiacAnalyzer:
    """
    Analyzes Left Ventricular (LV) function from contour data in an XML file.

    Calculates LV volume, Ejection Fraction (EF), and Global Longitudinal Strain (GLS)
    from time-resolved 2D endocardial contour points.

    Attributes:
        raw_contours (List[np.ndarray]): List of raw 49-point contours for each frame.
                                         Coordinates are in pixels.
        resampled_contours (List[np.ndarray]): List of 100-point resampled contours.
        times_sec (np.ndarray): Array of time points for each frame in seconds.
        pixel_size_mm (float): The size of a pixel in millimeters.
        volumes_ml (np.ndarray): Calculated LV volume for each frame in milliliters.
        gls_curve (np.ndarray): Calculated Global Longitudinal Strain (%) for each frame.
        L0 (float): The initial end-diastolic contour length in mm.
    """

    def __init__(self, xml_file_path: str):
        """
        Initializes the analyzer and parses the XML file.

        Args:
            xml_file_path: The path to the 'SpreadsheetML' (.xml) file.
        """
        self.xml_path = xml_file_path
        self._parse_xml_data()
        self.resampled_contours = []
        self.volumes_ml = np.array([])
        self.gls_curve = np.array([])
        self.L0 = 0.0

    def _parse_xml_data(self):
        """
        Parses the SpreadsheetML XML file to extract contour data and metadata.
        Handles the transposed layout of coordinates (X and Y in separate blocks).
        """
        # Namespace for SpreadsheetML
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {e}")

        all_rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', ns)
        
        # Temporary storage
        time_intervals_ms = []
        x_coords_by_point = []
        y_coords_by_point = []
        
        # State machine for parsing
        parsing_state = None 

        for row in all_rows:
            cells = row.findall('ss:Cell', ns)
            if not cells:
                continue

            first_cell_data = cells[0].find('ss:Data', ns)
            if first_cell_data is not None and first_cell_data.text:
                header = first_cell_data.text.strip()
                
                if header == 'frametime':
                    time_intervals_ms = [float(c.find('ss:Data', ns).text) for c in cells[1:] if c.find('ss:Data', ns) is not None]
                elif header == 'pixelsize':
                    self.pixel_size_mm = float(cells[1].find('ss:Data', ns).text)
                elif header == 'EndoX':
                    parsing_state = 'EndoX'
                    continue
                elif header == 'EndoY':
                    parsing_state = 'EndoY'
                    continue
            
            # Collect coordinate data based on current state
            if parsing_state == 'EndoX':
                # Stop if we hit the next section
                if first_cell_data is not None and first_cell_data.text == 'EndoY':
                    parsing_state = 'EndoY'
                    continue
                
                # Check if the row contains valid numbers
                row_data = [c.find('ss:Data', ns) for c in cells]
                if all(c is not None and c.get('{urn:schemas-microsoft-com:office:spreadsheet}Type') == 'Number' for c in row_data):
                     x_coords_by_point.append([float(c.text) for c in row_data])

            elif parsing_state == 'EndoY':
                if first_cell_data is not None and first_cell_data.text == 'EpiX':
                    parsing_state = None # End of EndoY
                    continue
                
                row_data = [c.find('ss:Data', ns) for c in cells]
                if all(c is not None and c.get('{urn:schemas-microsoft-com:office:spreadsheet}Type') == 'Number' for c in row_data):
                    y_coords_by_point.append([float(c.text) for c in row_data])
        
        if not time_intervals_ms or not x_coords_by_point or not y_coords_by_point:
            raise ValueError("Could not find 'frametime', 'EndoX', or 'EndoY' data in the XML.")

        # Convert to numpy arrays and transpose to get (num_frames, num_points)
        x_coords_by_frame = np.array(x_coords_by_point).T
        y_coords_by_frame = np.array(y_coords_by_point).T
        
        # Combine into a list of contours (one per frame)
        self.raw_contours = [np.column_stack((x, y)) for x, y in zip(x_coords_by_frame, y_coords_by_frame)]

        # Calculate cumulative time axis in seconds
        self.times_sec = np.cumsum([0] + time_intervals_ms) / 1000.0

    def _resample_contours(self, num_points: int = 100):
        """
        Resamples each raw contour to a fixed number of points using cubic splines.
        """
        self.resampled_contours = []
        for contour in self.raw_contours:
            # Add the first point to the end to ensure the spline is well-behaved at the closure
            points_to_fit = np.vstack([contour, contour[0]])
            
            # splprep finds the parametric representation of the curve
            # k=3 for cubic, s is a smoothing factor. s=0 means all points are interpolated.
            # A small s can help with numerical stability and prevent waviness.
            tck, u = splprep([points_to_fit[:, 0], points_to_fit[:, 1]], s=0, per=True)
            
            # splev evaluates the spline at new, equidistant parameter values
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck, der=0)
            
            self.resampled_contours.append(np.column_stack((x_new, y_new)))

    def _calculate_volumes(self):
        """
        Calculates the LV volume for each frame using the Method of Disks.
        
        This involves dynamically finding the apex, orienting the chamber, and summing
        the volumes of circular disks stacked along the long axis.
        """
        volumes_px3 = []
        for i, contour in enumerate(self.resampled_contours):
            # Use original raw contour for landmark identification for accuracy
            raw_contour = self.raw_contours[i]
            
            # 1. Define Mitral Base and Midpoint
            mitral_p1 = raw_contour[0]
            mitral_p48 = raw_contour[-1] # Point 49
            mitral_midpoint = (mitral_p1 + mitral_p48) / 2.0
            base_vector = mitral_p48 - mitral_p1
            
            # 2. Find Apex (point farthest from the mitral base line)
            distances = np.abs(np.cross(base_vector, raw_contour - mitral_p1)) / np.linalg.norm(base_vector)
            apex_index = np.argmax(distances)
            apex = raw_contour[apex_index]

            # 3. Define Long Axis
            long_axis_vector = apex - mitral_midpoint
            long_axis_length = np.linalg.norm(long_axis_vector)
            
            # 4. Orient the chamber (for resampled contour)
            # Translate so mitral midpoint is at origin
            translated_contour = contour - mitral_midpoint
            
            # Rotate so long axis aligns with y-axis
            angle = np.arctan2(long_axis_vector[0], long_axis_vector[1])
            rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], 
                                        [-np.sin(angle), np.cos(angle)]])
            oriented_contour = np.dot(rotation_matrix, translated_contour.T).T
            
            # 5. Method of Disks Calculation
            # Sort points by their y-coordinate (along the now-vertical long axis)
            sorted_indices = np.argsort(oriented_contour[:, 1])
            sorted_contour = oriented_contour[sorted_indices]

            volume = 0.0
            for j in range(len(sorted_contour) - 1):
                y1, y2 = sorted_contour[j, 1], sorted_contour[j+1, 1]
                r1, r2 = np.abs(sorted_contour[j, 0]), np.abs(sorted_contour[j+1, 0])
                
                h = y2 - y1
                # Use the average radius for the disk (frustum segment)
                avg_radius = (r1 + r2) / 2.0
                disk_volume = np.pi * (avg_radius**2) * h
                
                # Only add volume for slices along the long axis
                if h > 0 and y2 <= long_axis_length:
                    volume += disk_volume

            volumes_px3.append(volume)
        
        # Convert volume from pixels^3 to milliliters (mm^3 -> ml)
        volumes_px3 = np.array(volumes_px3)
        self.volumes_ml = volumes_px3 * (self.pixel_size_mm**3) / 1000.0

    def _calculate_ef(self) -> float:
        """Calculates Ejection Fraction from the volume curve."""
        if len(self.volumes_ml) == 0:
            raise RuntimeError("Volumes must be calculated before EF.")
        
        edv = np.max(self.volumes_ml)
        esv = np.min(self.volumes_ml)
        
        if edv == 0:
            return 0.0
            
        return ((edv - esv) / edv) * 100.0

    def _calculate_gls(self):
        """
        Calculates the Global Longitudinal Strain (GLS) curve.
        
        GLS is the percentage change in the total contour length relative to the
        end-diastolic length.
        """
        if not self.resampled_contours:
            raise RuntimeError("Contours must be resampled before GLS calculation.")
            
        # Calculate length of a single contour
        def contour_length(contour: np.ndarray) -> float:
            # Convert to mm first
            contour_mm = contour * self.pixel_size_mm
            # Sum of distances between adjacent points
            lengths = np.sqrt(np.sum(np.diff(contour_mm, axis=0)**2, axis=1))
            return np.sum(lengths)
        
        # Reference length is at ED (first frame)
        self.L0 = contour_length(self.resampled_contours[0])
        
        if self.L0 == 0:
            self.gls_curve = np.zeros(len(self.resampled_contours))
            return

        gls_values = []
        for contour in self.resampled_contours:
            L_t = contour_length(contour)
            strain = ((L_t - self.L0) / self.L0) * 100.0
            gls_values.append(strain)
            
        self.gls_curve = np.array(gls_values)

    def analyze(self) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline and returns the results.

        Returns:
            A dictionary containing key clinical metrics:
            - 'ejection_fraction': LV Ejection Fraction (%).
            - 'peak_global_longitudinal_strain': The most negative GLS value (%).
            - 'end_diastolic_volume_ml': Maximum volume in ml.
            - 'end_systolic_volume_ml': Minimum volume in ml.
            - 'time_seconds': Time points for the curves.
            - 'volume_ml_curve': LV volume curve over the cardiac cycle.
            - 'gls_percent_curve': GLS curve over the cardiac cycle.
        """
        print("Step 1: Resampling contours to 100 points...")
        self._resample_contours()
        
        print("Step 2: Calculating volume curve using Method of Disks...")
        self._calculate_volumes()
        
        print("Step 3: Calculating Ejection Fraction...")
        ef = self._calculate_ef()
        
        print("Step 4: Calculating Global Longitudinal Strain curve...")
        self._calculate_gls()
        
        peak_gls = np.min(self.gls_curve) if len(self.gls_curve) > 0 else 0.0
        
        print("\nAnalysis Complete.")
        
        results = {
            'ejection_fraction': ef,
            'peak_global_longitudinal_strain': peak_gls,
            'end_diastolic_volume_ml': np.max(self.volumes_ml) if len(self.volumes_ml) > 0 else 0.0,
            'end_systolic_volume_ml': np.min(self.volumes_ml) if len(self.volumes_ml) > 0 else 0.0,
            'time_seconds': self.times_sec,
            'volume_ml_curve': self.volumes_ml,
            'gls_percent_curve': self.gls_curve,
        }
        
        return results

# Add this function inside the CardiacAnalyzer class, replacing the previous version.

    def visualize_motion(self, use_resampled: bool = True, save_animation: bool = False, filename: str = 'lv_motion.gif'):
        """
        Creates an animation of the LV contour motion over the cardiac cycle.

        This is a powerful debugging tool to verify that the data parsing,
        contour resampling, and landmark identification are correct.

        Args:
            use_resampled: If True, animates the 100-point resampled contours.
                           If False, animates the original 49-point raw contours.
            save_animation: If True, saves the animation to a file instead of displaying it.
            filename: The name of the file to save the animation to (e.g., 'lv_motion.gif').
        """
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
        
        # Determine global plot limits to keep the axes stable during animation
        all_points = np.vstack(contours_to_show)
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        padding = (x_max - x_min) * 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect('equal', 'box')
        ax.set_title('LV Endocardial Contour Motion')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True)

        # Initialize plot elements that will be updated in each frame
        line, = ax.plot([], [], 'o-', lw=2, color='dodgerblue', markersize=3)
        apex_dot, = ax.plot([], [], '*', color='red', markersize=15, label='Apex')
        base_line, = ax.plot([], [], '-', color='limegreen', lw=3, label='Mitral Base')
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=12,
                            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.legend(loc='upper right')

        # init function for the animation
        def init():
            line.set_data([], [])
            apex_dot.set_data([], [])
            base_line.set_data([], [])
            time_text.set_text('')
            return line, apex_dot, base_line, time_text

        # update function called for each frame of the animation
        def update(i):
            contour = contours_to_show[i]
            # Close the loop for plotting
            plot_contour = np.vstack([contour, contour[0]])
            line.set_data(plot_contour[:, 0], plot_contour[:, 1])

            # Recalculate landmarks for this specific frame to verify logic
            raw_contour = self.raw_contours[i]
            mitral_p1 = raw_contour[0]
            mitral_p48 = raw_contour[-1]
            base_vector = mitral_p48 - mitral_p1
            
            # --- FIX for DeprecationWarning ---
            # Using a more robust method for distance calculation that avoids 2D cross product.
            norm_base_sq = np.dot(base_vector, base_vector)
            if norm_base_sq == 0: # Avoid division by zero if base points are identical
                distances = np.linalg.norm(raw_contour - mitral_p1, axis=1)
            else:
                # Project each point vector onto the base vector and find the perpendicular component
                vecs_to_points = raw_contour - mitral_p1
                t = np.dot(vecs_to_points, base_vector) / norm_base_sq
                projection = mitral_p1 + t[:, np.newaxis] * base_vector
                distances = np.linalg.norm(vecs_to_points - projection, axis=1)

            apex = raw_contour[np.argmax(distances)]

            # --- FIX for RuntimeError ---
            # set_data requires a sequence (list/array) even for a single point.
            apex_dot.set_data([apex[0]], [apex[1]])
            base_line.set_data([mitral_p1[0], mitral_p48[0]], [mitral_p1[1], mitral_p48[1]])
            
            # Update text annotation with frame info
            frame_info = (
                f"Frame: {i}\n"
                f"Time: {self.times_sec[i]:.3f} s\n"
                f"Volume: {self.volumes_ml[i]:.2f} ml\n"
                f"GLS: {self.gls_curve[i]:.2f} %"
            )
            time_text.set_text(frame_info)
            
            return line, apex_dot, base_line, time_text

        # Create the animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(contours_to_show),
            init_func=init,
            blit=True,
            interval=50  # milliseconds between frames
        )

        if save_animation:
            try:
                # Pillow is a good writer for GIFs
                print(f"Saving animation to '{filename}'... This may take a moment.")
                anim.save(filename, writer='pillow', fps=20)
                print("Animation saved successfully.")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("You might need to install an animation writer. Try: pip install Pillow")
        else:
            plt.show()
			
# --- Example Usage ---
if __name__ == "__main__":
    xml_file = "validation.xml"
    
    try:
        analyzer = CardiacAnalyzer(xml_file)
        analysis_results = analyzer.analyze()
        
        print("\n--- Results ---")
        print(f"Ejection Fraction (EF): {analysis_results['ejection_fraction']:.2f} %")
        print(f"Peak Global Longitudinal Strain (GLS): {analysis_results['peak_global_longitudinal_strain']:.2f} %")
        print(f"End-Diastolic Volume (EDV): {analysis_results['end_diastolic_volume_ml']:.2f} ml")
        print(f"End-Systolic Volume (ESV): {analysis_results['end_systolic_volume_ml']:.2f} ml")
        print("-----------------\n")

        # --- NEW: Call the visualization function ---
        print("Starting visualization of the cardiac cycle motion.")
        # Animate the resampled contours used for calculation
        analyzer.visualize_motion(use_resampled=True)

        # Optional: You can also visualize the raw points to check the input data
        # print("Starting visualization of the raw contour points.")
        # analyzer.visualize_motion(use_resampled=False)
        
        # Optional: Save the animation as a GIF
        # analyzer.visualize_motion(use_resampled=True, save_animation=True, filename='lv_cycle.gif')

        # Optional: Plotting the static result curves
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Volume Curve
            ax1.plot(analysis_results['time_seconds'], analysis_results['volume_ml_curve'], 'o-', label='LV Volume')
            ax1.set_ylabel('Volume (ml)')
            ax1.set_title('LV Volume and Global Longitudinal Strain')
            ax1.grid(True)
            ax1.legend()
            
            # GLS Curve
            ax2.plot(analysis_results['time_seconds'], analysis_results['gls_percent_curve'], 'o-', color='r', label='GLS')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Strain (%)')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not installed. Cannot plot static curves.")

    except FileNotFoundError:
        print(f"Error: The file '{xml_file}' was not found.")
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"An error occurred: {e}")