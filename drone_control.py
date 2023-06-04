from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
from threading import Thread
from RopeDetector import RopeDetector

# Speed of the drone
S = 20
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

Z_AXIS_MOVE_UP = 1
Z_AXIS_MOVE_DOWN = -1
Y_AXIS_MOVE_RIGHT = 1
Y_AXIS_MOVE_LEFT = -1


class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - Esc: Quit
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
            - G: Enable rope following
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = target=Tello()

        self.detector = RopeDetector()
        self.rope_following = False

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        self.pic_number = 0

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):

        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()
        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    Thread(target = self.update()).start()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    elif event.key == pygame.K_g: # Toggle rope following
                        self.rope_following = not self.rope_following
                        if self.rope_following: # If rope following on, set forward velocity
                            self.for_back_velocity = 10
                        else: # If rope following off, make all velocities 0
                            self.left_right_velocity = 0
                            self.up_down_velocity = 0
                            self.for_back_velocity = 0

                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame

            # Perform rope detection 
            frame, movements = self.detector.detect_rope(frame)
            if self.rope_following and movements:
                self.for_back_velocity = 20

                # Update Y axis velocities
                if movements[0] == Y_AXIS_MOVE_LEFT:
                    self.left_right_velocity = 10
                elif movements[0] == Y_AXIS_MOVE_RIGHT:
                    self.left_right_velocity = -10
                else:
                    self.left_right_velocity = 0

                # Update Z axis velocities
                if movements[1] == Z_AXIS_MOVE_DOWN:
                    self.up_down_velocity = -10
                elif movements[1] == Z_AXIS_MOVE_UP:
                    self.up_down_velocity = 10   
                else:
                    self.up_down_velocity = 0
            
            # If no movements required, set velocities to 0
            elif self.rope_following and movements is None:
                self.left_right_velocity = 0
                self.up_down_velocity = 0
                self.for_back_velocity = 0
                

            # Show the frame to the operator so they can see what the algorithm is doing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = np.rot90(frame) 
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        # 通常在结束前调用它以释放资源
        self.tello.end()

    def keydown(self, key):
        """ 
            Update velocities based on key pressed
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_p: # Save the current frame
            frame = self.tello.get_frame_read()
            cv2.imwrite(f"picture{self.pic_number}.png", frame.frame)
            self.pic_number += 1
        elif key == pygame.K_g:
            self.tello.flip_left()

    def keyup(self, key):
        """ 
            Update velocities based on key released
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ 
            Send velocities to Tello.
        """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

def main():
    frontend = FrontEnd()
    frontend.run()

if __name__ == '__main__':
    main()
