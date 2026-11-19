import os
import threading
from time import monotonic

import cv2


class Webcam:
    def __init__(self):
        self.control = threading.RLock()
        self.latest = None
        self.sequence = 0
        self.thread = self.analysis_thread = None
        self.capture_stop = self.infer_stop = None
        self.result = None
        self.error = None

    def start(self,index,width):
        with self.control:
            self.stop()
            self.latest = self.result = self.error = None
            self.capture_stop = threading.Event()
            self.thread = threading.Thread(target=self._read,args=(index,width,self.capture_stop),daemon=True)
            self.thread.start()

    def _read(self,index,width,stop):
        cap = cv2.VideoCapture(index,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
        try:
            if not cap.isOpened():
                self.error = 'Não foi possível abrir a webcam.'
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,round(width*0.75))
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            while not stop.is_set():
                ok,frame = cap.read()
                if not ok:
                    self.error = 'A câmera não entregou um quadro.'
                    stop.wait(.08)
                    continue
                self.error = None
                ok,jpeg = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,82])
                if ok:
                    self.sequence += 1
                    self.latest = self.sequence,jpeg.tobytes()
        finally:
            cap.release()

    def stop(self):
        with self.control:
            if self.capture_stop:
                self.capture_stop.set()
            self.stop_analysis()
            if self.thread:
                self.thread.join(timeout=3)
                if self.thread.is_alive():
                    raise ValueError('A webcam ainda está encerrando. Tente novamente em instantes.')
            self.latest = None

    def stop_analysis(self):
        with self.control:
            if self.infer_stop:
                self.infer_stop.set()

    def continuous(self, seconds, operation):
        with self.control:
            if not self.thread or not self.thread.is_alive():
                raise ValueError('Inicie a webcam primeiro.')
            if self.analysis_thread and self.analysis_thread.is_alive():
                raise ValueError('A análise anterior ainda está em andamento.')
            self.infer_stop = stop = threading.Event()
            self.result = None
            def worker():
                last = None
                while not stop.is_set():
                    tick = monotonic()
                    frame = self.latest
                    if frame and frame[0] != last:
                        last = frame[0]
                        result = operation(frame[1])
                        if not stop.is_set():
                            self.result = last,result
                    stop.wait(max(.05,seconds-(monotonic()-tick)))
            self.analysis_thread = threading.Thread(target=worker,daemon=True)
            self.analysis_thread.start()
