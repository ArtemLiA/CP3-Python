"""
Package responsible for visualization of boids model simulation.    
"""

import numpy as np
import json

from simulation import *

from vispy import app, scene
from vispy.geometry import Rect
from vispy.scene.visuals import Text
from vispy.scene.visuals import Rectangle

import imageio as iio


class Simulation:
    """
    Class for visualization and recording simulation of boids.
    """
    def __init__(self, path, n_boids, dt = 1e-2):
        """
        :param str path: path to json-file with simulation parameters.
        :param int n_boids: number of boids.
        :param float dt: time step, defaults to 1e-2
        """
        with open(path) as file:
            config = json.load(file)
        
        self.width = config["window"]["width"]
        self.height = config["window"]["height"]
        self.asp = self.width/self.height
        
        a = config["coefficients"]["alignment"]
        c = config["coefficients"]["cohesion"]
        s = config["coefficients"]["separation"]
        w = config["coefficients"]["walls"]
        n = config["coefficients"]["noise"]
        
        self.alignment = a
        self.cohesion = c
        self.separation = s
        self.walls = w
        self.noise = n
        
        self.coefficients = np.array([a, c, s, w, n], dtype=np.double)
        
        self.alpha = config["visibility"]["alpha"]
        self.perception = config["visibility"]["perception"]
        
        self.v_range = tuple(config["v_range"])
        self.w_range = tuple(config["w_range"])
        
        self.boids = init_boids(n_boids, self.asp, self.v_range)
        self.n_boids = n_boids
        self.dt = dt
        
        # Id of selected boid
        self.id = np.random.choice(self.n_boids)
        
        self._create_visual_tools()
        
        self.frame = 0
    
    def _create_visual_tools(self):
        """
        Creates VisPy tools to visualize simulation.
        """
        self.canvas = scene.SceneCanvas(show=True,
                                        size=(self.width, self.height),
                                        resizable=False)   
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=Rect(0, 0, self.asp, 1))
        
        self._create_arrows()
        self._create_visible_area()
        self._create_text()
    
    def _create_arrows(self):
        """
        Creates arrows to visualize boids.
        """
        self.arrows = scene.Arrow(
            arrow_color = (255/255, 248/255, 0/255, 1),
            arrow_size = 8,
            connect = "segments",
            parent = self.view.scene
        )
        
        self.visible_arrows = scene.Arrow(
            arrow_color = (255/255, 0/255, 0/255, 1),
            arrow_size = 8,
            connect = "segments",
            parent = self.view.scene
        )
        
        self.selected_arrow = scene.Arrow(
            arrow_color = (255/255, 0/255, 231/255, 1), 
            arrow_size = 8,
            connect = "segments",
            parent = self.view.scene
        )
    
    def _create_visible_area(self):
        """
        Creates ellipse for visualization of chosen boid's visibility area.
        """
        self.visible_area = scene.Ellipse(
            center = self.boids[self.id, 0:2],
            color = (255/255, 0/255, 215/255, 0.22),
            radius = self.perception,
            border_width = 0,
            parent = self.view.scene
        )
    
    def _create_text(self):
        """
        Creates a windom (rectangle) with text.
        """
        font_size = 8
        row_height = 1/36
        num_rows = 7  # num_boids, fps, 5 coefficients

        rect_width = 1/6
        rect_height = num_rows * row_height
        x_center, y_center = rect_width/2, 1.0 - rect_height/2
        
        self.text_rect = Rectangle(
            center = (x_center, y_center),
            color = (1, 1, 1, 0.85),
            width = rect_width,
            height = rect_height,
            parent = self.view.scene
        )
        
        y_centers = {}
        for i in range(1, num_rows + 1):
            y_centers[i] = 1.0 - (i - 0.5) * row_height
        
        self.text_fps = Text(
            text = f"FPS: {self.canvas.fps:.1f}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[1]],
            font_size = font_size,
            parent = self.view.scene
        )
        
        self.text_num_boids = Text(
            text = f"Boids: {self.n_boids}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[2]],
            font_size = font_size,
            parent = self.view.scene
        )
        
        self.text_alignment = Text(
            text = f"Alignment: {self.alignment:.2f}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[3]],
            font_size = font_size,
            parent = self.view.scene
        )
        
        self.text_cohesion = Text(
            text = f"Cohesion: {self.cohesion:.2f}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[4]],
            font_size = font_size,
            parent = self.view.scene
        )
        
        self.text_separation = Text(
            text = f"Separation: {self.separation:.2f}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[5]],
            font_size = font_size,
            parent = self.view.scene    
        )
        
        self.text_walls = Text(
            text = f"Walls: {self.walls:.2f}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[6]],
            font_size = font_size,
            parent = self.view.scene
        )
        
        self.text_noise = Text(
            text = f"Noise: {self.noise:.2f}",
            color = (0, 0, 0, 1),
            pos = [x_center, y_centers[7]],
            font_size = font_size,
            parent = self.view.scene
        )

    def update_video(self):
        """
        Records a new frame in the video or completes recoding.
        """
        if self.writer is None:
           return
    
        if self.frame <= self.video_frames:
            frame = self.canvas.render(alpha=False)
            self.writer.append_data(frame)
        else:
            self.writer.close()
            self.writer = None
            self.video_frames = None
            print("The video was recorded successfully!")
            
            if self.close_app:
                app.quit()
  
    def update(self, event):
        """
        Updates boids and render boids model visualization on screen 
        using VisPy tools.

        :param vispy.util.event.Event event: event object.
        """
        self._update_boids()
        self._update_arrows()
        self._update_area()
        self._update_text()
        
        self.frame += 1
        self.update_video()
           
    def _update_arrows(self):
        """
        Updates arrows.
        """
        boids = self.boids
        boids_directions = get_directions(self.boids, self.dt)
        
        self.arrows.set_data(arrows=boids_directions)
        self.selected_arrow.set_data(arrows=boids_directions[[self.id]])
        
        mask = visibility_mask(boids, self.id, self.alpha, self.perception)
        self.visible_arrows.set_data(arrows=boids_directions[mask])
    
    def _update_area(self):
        """
        Updates visibility area of chosen boid.
        """
        self.visible_area.center = self.boids[self.id, 0:2]
    
    def _update_text(self):
        """
        Updates text on screen.
        """
        if self.frame % 30 == 0:
            self.text_fps.text = f"FPS: {self.canvas.fps:.1f}"
            self.canvas.fps    
    
    def _update_boids(self):
        """
        Implements the step of a complete update of boids model.
        """
        flocking(self.boids, self.alpha, self.perception, 
                 self.coefficients, self.asp, self.v_range)
        propagate(self.boids, self.dt, self.v_range, self.w_range)
        walls_collision(self.boids, self.asp)
             
    def start(self, record_video = False, fps = 60, duration = 35, file = None,
              close_app = False):
        """
        Starts boids model simulation and drawing it on the screen.

        :param bool record_video: if true, then will be recorded video, if
            False simulation will only be shown on screen, defaults to False.
        :param int fps: fps of recorded video, defaults to 60.
        :param int duration: duration in seconds of recorded video, 
            defaults to 35
        :param str file: name of file to record simulation, if None then
            simulation will be recorded to file 'video_{n_boids}.mp4', 
            defaults to None
        :param bool close_app: if True then VisPy app will be automatically
            closed after ending of recording video, defaults to False.
        """
        self.close_app = close_app
        self.video_frames = None
        self.writer = None
        
        if record_video: 
            file = f"video_{self.n_boids}.mp4" if file is None else file
            self.video_frames = fps * duration
            self.writer = iio.get_writer(file, fps=fps)
        
        self.timer = app.Timer(interval=0, start=True, connect=self.update)
        self.canvas.measure_fps()
        app.run()
        
        
if __name__ == "__main__":
    sim_1000 = Simulation(path="config.json", n_boids=1_000)
    sim_1000.start(record_video=True, fps=60, duration=50, close_app=True)
    
    sim_5000 = Simulation(path="config.json", n_boids=5_000)
    sim_5000.start(record_video=True, fps=60, duration=35, close_app=True)   
