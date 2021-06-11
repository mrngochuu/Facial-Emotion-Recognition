import sys
import cv2
import math
from threading import Thread

if sys.version_info >= (3, 0):
    from queue import Queue

else:
    from Queue import Queue


class FileVideosStreamCustom:
  
  def __init__(self, path, queueSize = 128, frameLimit=0):
      # initialize the file video stream along with the boolean
  # used to indicate if the thread should be stopped or not
    self.stream = cv2.VideoCapture(path)
    self.stopped = False
    # initialize the queue used to store frames read from
    # the video file
    self.Q = Queue(maxsize=queueSize)
    # fps
    self.fps = self.stream.get(cv2.CAP_PROP_FPS)
    self.totalFrame = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
    self.frameLimit = frameLimit
    self.fpsFlg = False

  def start(self):
    self.reduceFPS()

    # start a thread to read frames from the file video stream
    t = Thread(target=self.update, args=())
    t.daemon = True
    t.start()
    return self

  def reduceFPS(self):
    if self.frameLimit < self.fps and self.frameLimit != 0:
      self.fpsFlg = True
    
  def update(self):
    # keep looping infinitely
    # flg to cut frame
    flg = True
    while True:
        # if the thread indicator variable is set, stop the
        # thread
        if self.stopped:
            return
        # otherwise, ensure the queue has room in it
        if not self.Q.full():
            # read the next frame from the file
            (grabbed, frame) = self.stream.read()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return

            # apply fps
            if self.fpsFlg:
              currentFrame = self.stream.get(cv2.CAP_PROP_POS_FRAMES)
              
              if flg:
                firstFrame = currentFrame
                lastFrame = firstFrame + int(self.fps)

              if currentFrame >= firstFrame and currentFrame <= firstFrame + self.frameLimit:
                flg = False
                self.Q.put(frame)
              elif currentFrame == lastFrame:
                flg = True
            else:
              # add the frame to the queue
              self.Q.put(frame)

  def read(self):
    # return next frame in the queue
    return self.Q.get()

  def stop(self):
  # indicate that the thread should be stopped
    self.stopped = True