import picamera
import time

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()

    for i in range(3):
        time.sleep(2)
        camera.capture('test{0}.jpg'.format(i))
