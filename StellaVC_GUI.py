import os
import sys
import wave
import pyaudio
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtMultimedia
import threading
import time
import shutil
import StellaVC
import utils
import warnings

warnings.filterwarnings('ignore')

class EmitStr(QObject):
    textWrite = pyqtSignal(str)

    def write(self, text):
        self.textWrite.emit(str(text))


class BaseGuiWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        sys.stdout = EmitStr(textWrite=self.outputWrite)  # redirect stdout
        sys.stderr = EmitStr(textWrite=self.outputWrite)  # redirect stderr

        self.initUI()

    def initUI(self):
        self.resize(1600, 800)
        self.setFixedSize(1600, 800)
        self.center()
        self.setWindowTitle('Stella Voice Changer')
        self.setObjectName('MainWindow')
        self.setWindowIcon(QIcon('assets/icon-nat.ico'))
        self.setStyleSheet('#MainWindow{border-image:url(assets/bg-init.png)}')

        self.resource_dir = None # 模型特有的资源文件夹

        col = QColor(245, 245, 245)

        # ------------------- #
        # configuration panel #
        # ------------------- #
        self.configFrame = QFrame(self)
        self.configFrame.setFrameShape(QFrame.StyledPanel)
        self.configFrame.setStyleSheet("QWidget {background-color: rgba(245, 245, 245, 200)}")
        self.configFrame.setWindowOpacity(0.6)
        self.configFrame.setGeometry(20, 20, 400, 280)

        self.hubertButton = QPushButton('hubert path', self.configFrame)
        self.hubertButton.setFixedSize(100, 50)
        self.hubertButton.move(10, 10)
        self.hubertButton.clicked.connect(lambda: self.chooseFile('pt'))
        self.vitsButton = QPushButton('vits path', self.configFrame)
        self.vitsButton.setFixedSize(100, 50)
        self.vitsButton.move(10, 80)
        self.vitsButton.clicked.connect(lambda: self.chooseFile('pth'))
        self.configButton = QPushButton('config path', self.configFrame)
        self.configButton.setFixedSize(100, 50)
        self.configButton.move(10, 150)
        self.configButton.clicked.connect(lambda: self.chooseFile('json'))
        self.loadButton = QPushButton('load model', self.configFrame)
        self.loadButton.setFixedSize(380, 50)
        self.loadButton.move(10, 220)
        self.loadButton.clicked.connect(self.loadModel)

        self.hubertPath = QTextEdit(self.configFrame)
        self.hubertPath.setFocusPolicy(Qt.NoFocus)
        self.hubertPath.setFixedSize(270, 50)
        self.hubertPath.move(120, 10)
        self.vitsPath = QTextEdit(self.configFrame)
        self.vitsPath.setFocusPolicy(Qt.NoFocus)
        self.vitsPath.setFixedSize(270, 50)
        self.vitsPath.move(120, 80)
        self.configPath = QTextEdit(self.configFrame)
        self.configPath.setFocusPolicy(Qt.NoFocus)
        self.configPath.setFixedSize(270, 50)
        self.configPath.move(120, 150)

        self.hubert_path = ''
        self.vits_path = ''
        self.config_path = ''

        self.thread_sovits = threading.Thread(target=lambda: StellaVC.voice_conversion(self.hubert_path,
                                                                      self.vits_path,
                                                                      self.config_path))

        # ----------------- #
        # information panel #
        # ----------------- #
        self.infoFrame = QFrame(self)
        self.infoFrame.setFrameShape(QFrame.StyledPanel)
        self.infoFrame.setStyleSheet("QWidget {background-color: rgba(245, 245, 245, 200)}")
        self.infoFrame.setGeometry(20, 320, 400, 460)

        self.infoBox = QTextEdit(self.infoFrame)
        self.infoBox.setFocusPolicy(Qt.NoFocus)
        self.infoBox.setFixedSize(390, 450)
        self.infoBox.move(5, 5)

        # ---------- #
        # main panel #
        # ---------- #
        self.mainFrame = QFrame(self)
        self.mainFrame.setFrameShape(QFrame.StyledPanel)
        self.mainFrame.setStyleSheet("QWidget {background-color: rgba(245, 245, 245, 200)}")
        self.mainFrame.setGeometry(440, 20, 1140, 760)

        # GPU mode
        self.gpuButton = QPushButton('', self.mainFrame)
        self.gpuButton.setStyleSheet("QPushButton {border-image: url(assets/gpu-off.png); background-color: transparent}")
        self.gpuButton.setFixedSize(80, 40)
        self.gpuButton.move(530, 20)
        self.gpuButton.clicked.connect(self.useGPU)
        self.gpu_mode = False


        # character portrait
        self.charFrame = QLabel(self.mainFrame)
        self.charFrame.setFrameShape(QFrame.StyledPanel)
        self.charFrame.resize(256, 256)
        self.charFrame.move(442, 80)

        self.charPortrait = QPixmap('assets/portrait-init.png')

        self.charFrame.setPixmap(self.charPortrait)

        # TODO: 支持多人模型
        self.charCombo = QComboBox(self.mainFrame)
        self.charCombo.setGeometry(442, 350, 256, 30)
        self.charCombo.currentIndexChanged.connect(lambda: self.chooseChar(self.charCombo.currentIndex()))

        # record
        self.record_path = 'sovits_cache/record.wav'
        self.micLabel = QLabel('', self.mainFrame)
        self.micLabel.setStyleSheet("QLabel {border-image: url(assets/microphone.png); background-color: transparent}")
        self.micLabel.setFixedSize(100, 100)
        self.micLabel.move(171, 150) # cp.x: 221
        self.recordLabel = QLabel('', self.mainFrame)
        self.recordLabel.setStyleSheet("QLabel {border-image: url(assets/from-mic-1.png); background-color: transparent}")
        self.recordLabel.setFixedSize(100, 100)
        self.recordLabel.move(311, 150)
        self.recordTimeLabel = QLabel('00:00', self.mainFrame)
        self.recordTimeLabel.setStyleSheet("QLabel {background-color: transparent}")
        self.recordTimeLabel.setFixedSize(40, 40)
        self.recordTimeLabel.move(201, 240)
        self.startRecordButton = QPushButton('', self.mainFrame)
        self.stopRecordButton = QPushButton('', self.mainFrame)
        self.startRecordButton.setStyleSheet("QPushButton {border-image: url(assets/start-record.png); background-color: transparent}")
        self.stopRecordButton.setStyleSheet("QPushButton {border-image: url(assets/stop-record-unable.png); background-color: transparent}")
        self.startRecordButton.setFixedSize(50, 50)
        self.stopRecordButton.setFixedSize(50, 50)
        self.startRecordButton.move(161, 280)
        self.stopRecordButton.move(231, 280)
        self.startRecordButton.clicked.connect(self.startRecord)
        self.stopRecordButton.clicked.connect(self.stopRecord)

        self.recordTimer = QTimer()
        self.recordTimer.start(1000)
        self.recordTimer.timeout.connect(self.updateTimer)
        self.recordSec = 0

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.is_recording = False # 是否正在录制
        self.event_start_record = threading.Event()
        self.thread_record = threading.Thread(target=self.recordAudio)
        self.thread_record.setDaemon(True)
        self.thread_record.start()

        # upload
        self.uploadButton = QPushButton('', self.mainFrame)
        self.uploadButton.setStyleSheet("QPushButton {border-image: url(assets/upload.png); background-color: transparent}")
        self.uploadButton.setFixedSize(100, 100)
        self.uploadButton.move(869, 150) # cp.x: 919
        self.uploadButton.clicked.connect(self.uploadAudio)
        self.uploadLabel = QLabel('', self.mainFrame)
        self.uploadLabel.setStyleSheet("QLabel {border-image: url(assets/from-upload-1.png); background-color: transparent}")
        self.uploadLabel.setFixedSize(100, 100)
        self.uploadLabel.move(729, 150)

        # play
        self.playFrame = QFrame(self.mainFrame)
        self.playFrame.setFixedSize(1100, 200)
        self.playFrame.move(20, 540)
        self.playFrame.setFrameShape(QFrame.Box)
        self.playFrame.setFrameShadow(QFrame.Raised)
        self.playFrame.setStyleSheet("QFrame {border-width: 3px; border-style: solid; border-color: rgb(18, 150, 219); background-color: transparent}")

        # convert
        self.save_path = 'sovits_cache/temp.wav'
        self.convertButton = QPushButton('', self.mainFrame)
        self.convertButton.setStyleSheet("QPushButton {border-image: url(assets/start.png); background-color: transparent}")
        self.convertButton.setFixedSize(50, 50)
        self.convertButton.move(440, 605)
        self.convertButton.clicked.connect(self.convertAudio)

        self.play_path = 'sovits_cache/play.wav'
        self.playButton = QPushButton('', self.mainFrame)
        self.playButton.setFixedSize(100, 100)
        self.playButton.move(520, 580)
        self.playButton.setStyleSheet("QPushButton {border-image: url(assets/play.png); background-color: transparent}")
        self.playButton.clicked.connect(self.playAudio)
        self.playFlag = False
        self.playThread = threading.Thread(target=self.setCurrentPlaying)
        self.playThread.setDaemon(True)

        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(50)

        self.startTimeLabel = QLabel('00:00', self.mainFrame)
        self.endTimeLabel = QLabel('00:00', self.mainFrame)
        self.startTimeLabel.setStyleSheet("QLabel {background-color: transparent}")
        self.endTimeLabel.setStyleSheet("QLabel {background-color: transparent}")
        self.startTimeLabel.setFixedSize(40,40)
        self.endTimeLabel.setFixedSize(40, 40)
        self.startTimeLabel.move(70, 687)
        self.endTimeLabel.move(1030, 687)
        self.slider = QSlider(Qt.Horizontal, self.mainFrame)
        self.slider.setFixedSize(800, 20)
        self.slider.move(170, 700)

        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.updateSlider)
        self.slider.sliderMoved[int].connect(lambda: self.player.setPosition(self.slider.value()))

        # download
        self.download_path = 'sovits_cache/download.wav'
        self.downloadButton = QPushButton('', self.mainFrame)
        self.downloadButton.setStyleSheet("QPushButton {border-image: url(assets/download.png); background-color: transparent}")
        self.downloadButton.setFixedSize(50, 50)
        self.downloadButton.move(650, 605)
        self.downloadButton.clicked.connect(self.downloadAudio)

        # change mode
        self.modeButton = QPushButton('', self.mainFrame)
        self.modeButton.setStyleSheet("QPushButton {border-image: url(assets/mode-hubert.png); background-color: transparent}")
        self.modeButton.setFixedSize(50, 50)
        self.modeButton.move(360, 605)
        self.mode = False # False为hubert，True为flow
        self.modeButton.clicked.connect(self.changeMode)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def chooseFile(self, file_type):
        if file_type == 'pt':
            file_path = QFileDialog.getOpenFileName(self, f'Choose hubert model', '/home', f'(*.pt)')
            if file_path[0]:
                self.hubertPath.setText(file_path[0])
                self.hubert_path = file_path[0]
        elif file_type == 'pth':
            file_path = QFileDialog.getOpenFileName(self, f'Choose vits model', '/home', f'(*.pth)')
            if file_path[0]:
                self.vitsPath.setText(file_path[0])
                self.vits_path = file_path[0]
        elif file_type == 'json':
            file_path = QFileDialog.getOpenFileName(self, f'Choose configuration file', '/home', f'(*.json)')
            if file_path[0]:
                self.configPath.setText(file_path[0])
                self.config_path = file_path[0]
        else:
            raise ValueError('Unsupported file type!')

    # TODO: 支持GPU推理
    def useGPU(self):
        QMessageBox.warning(self,
                             'Warning',
                             'GPU mode is not supported yet!'
                             )
        # if not self.gpu_mode:
        #     reply = QMessageBox.question(self,
        #                                  'Question',
        #                                  'Are you sure to use GPU mode?',
        #                                  QMessageBox.Yes | QMessageBox.No,
        #                                  QMessageBox.No)
        #     if reply == QMessageBox.Yes:
        #         print('GPU mode! Please reload the models!')
        #         self.gpu_mode = True
        #         self.gpuButton.setStyleSheet("QPushButton {border-image: url(assets/gpu-on.png); background-color: transparent}")
        #     else:
        #         pass
        # else:
        #     reply = QMessageBox.question(self,
        #                                  'Question',
        #                                  'Are you sure to use CPU mode?',
        #                                  QMessageBox.Yes | QMessageBox.No,
        #                                  QMessageBox.No)
        #     if reply == QMessageBox.Yes:
        #         print('GPU mode! Please reload the models!')
        #         self.gpu_mode = False
        #         self.gpuButton.setStyleSheet("QPushButton {border-image: url(assets/gpu-off.png); background-color: transparent}")
        #     else:
        #         pass

    # 根据配置文件改变图片
    def chooseChar(self, index=0):
        self.charPortrait.load(os.path.join(self.resource_dir, f'{index}.png'))
        self.charFrame.setPixmap(self.charPortrait)
        StellaVC.select_speaker(index)

    def loadModel(self):
        if self.hubert_path and self.hubert_path and self.config_path:
            while self.thread_sovits.is_alive():
                StellaVC.terminate_vc()

            # 声音转换线程（结束之前的线程后新注册一个线程）
            self.thread_sovits = threading.Thread(target=lambda: StellaVC.voice_conversion(self.hubert_path,
                                                                                           self.vits_path,
                                                                                           self.config_path))
            self.thread_sovits.setDaemon(True) # 防止主线程结束后子线程仍然运行
            self.thread_sovits.start()

            # 选择speaker面板加载
            hps = utils.get_hparams_from_file(self.config_path)
            model_name = hps.info.model_name
            speakers = hps.info.speakers
            self.resource_dir = f'assets/figures/{model_name}'

            self.setStyleSheet('#MainWindow {border-image:url(%s)}'
                               % f'{self.resource_dir}/bg.png')

            self.charCombo.clear()
            for speaker in speakers:
                self.charCombo.addItem(speaker)

            self.charPortrait.load(os.path.join(self.resource_dir, '0.png'))
            self.charFrame.setPixmap(self.charPortrait)

            # single speaker
            if len(speakers) == 1:
                self.charCombo.setDisabled(True)
            else:
                self.charCombo.setDisabled(False)



    def outputWrite(self, text):
        self.infoBox.append(text)

    # 连接到麦克风录音
    def recordAudio(self):
        while True:
            self.event_start_record.wait()
            self.event_start_record.clear()

            print('Start recording...')
            p = pyaudio.PyAudio()
            stream =p.open(format=self.FORMAT,
                           channels=self.CHANNELS,
                           rate=self.RATE,
                           frames_per_buffer=self.CHANNELS,
                           input=True)
            frames = []
            while self.is_recording:
                data = stream.read(self.CHUNK)
                frames.append(data)

            print('Stop recording...')
            stream.stop_stream()  # 停止錄音
            stream.close()  # 關閉串流
            p.terminate()

            # 保存录音文件到cache
            wave_file = wave.open(self.record_path, 'wb')
            wave_file.setnchannels(self.CHANNELS)  # 設定聲道
            wave_file.setsampwidth(p.get_sample_size(self.FORMAT))  # 設定格式
            wave_file.setframerate(self.RATE)  # 設定取樣頻率
            wave_file.writeframes(b''.join(frames))  # 存檔
            wave_file.close()

            # 将录音文件加载到模型中
            StellaVC.load_audio(self.record_path)

    # 更新录音时的计时器
    def updateTimer(self):
        if self.is_recording:
            self.recordSec += 1
            self.recordTimeLabel.setText(time.strftime('%M:%S', time.gmtime(self.recordSec)))

    def startRecord(self):
        self.recordSec = 0
        self.startRecordButton.setDisabled(True)
        self.startRecordButton.setStyleSheet("QPushButton {border-image: url(assets/start-record-unable.png); background-color: transparent}")
        self.stopRecordButton.setDisabled(False)
        self.stopRecordButton.setStyleSheet("QPushButton {border-image: url(assets/stop-record.png); background-color: transparent}")
        self.uploadButton.setDisabled(False)
        self.uploadButton.setStyleSheet("QPushButton {border-image: url(assets/upload-unable.png); background-color: transparent}")
        self.uploadLabel.setStyleSheet("QLabel {border-image: url(assets/from-upload-unable.png); background-color: transparent}")
        self.is_recording = True
        self.event_start_record.set()

    def stopRecord(self):
        self.startRecordButton.setDisabled(False)
        self.startRecordButton.setStyleSheet("QPushButton {border-image: url(assets/start-record.png); background-color: transparent}")
        self.stopRecordButton.setDisabled(True)
        self.stopRecordButton.setStyleSheet("QPushButton {border-image: url(assets/stop-record-unable.png); background-color: transparent}")
        self.uploadButton.setDisabled(False)
        self.uploadButton.setStyleSheet("QPushButton {border-image: url(assets/upload.png); background-color: transparent}")
        self.uploadLabel.setStyleSheet("QLabel {border-image: url(assets/from-upload-1.png); background-color: transparent}")
        self.is_recording = False

    def uploadAudio(self):
        self.startRecordButton.setDisabled(True)
        self.startRecordButton.setStyleSheet("QPushButton {border-image: url(assets/start-record-unable.png); background-color: transparent}")
        self.recordLabel.setStyleSheet("QLabel {border-image: url(assets/from-mic-unable.png); background-color: transparent}")
        self.micLabel.setStyleSheet("QLabel {border-image: url(assets/microphone-unable.png); background-color: transparent}")
        file_path = QFileDialog.getOpenFileName(self, 'Choose source audio file', '/home', '(*.wav *mp3 *ogg *flv)')
        if file_path[0]:
            StellaVC.load_audio(file_path[0])

        self.startRecordButton.setDisabled(False)
        self.startRecordButton.setStyleSheet("QPushButton {border-image: url(assets/start-record.png); background-color: transparent}")
        self.recordLabel.setStyleSheet("QLabel {border-image: url(assets/from-mic-1.png); background-color: transparent}")
        self.micLabel.setStyleSheet("QLabel {border-image: url(assets/microphone.png); background-color: transparent}")

    def convertAudio(self):
        self.player.setMedia(QtMultimedia.QMediaContent()) # 解除占用
        StellaVC.convert_audio()

    # 播放音频
    def setCurrentPlaying(self):
        try:
            content = QtMultimedia.QMediaContent(QUrl.fromLocalFile(self.play_path))
            self.player.setMedia(content)
        except:
            print('Nothing to play!')

    def playAudio(self):
        if not self.player.isAudioAvailable():
            self.setCurrentPlaying()
        if not self.playFlag:
            self.convertButton.setDisabled(True)
            self.convertButton.setStyleSheet("QPushButton {border-image: url(assets/start-unable.png); background-color: transparent}}")
            self.playFlag = True
            #self.playButton.setText('pause')
            self.playButton.setStyleSheet("QPushButton {border-image: url(assets/pause.png); background-color: transparent}}")
            self.player.play()
        else:
            self.playFlag = False
            #self.playButton.setText('play')
            self.playButton.setStyleSheet("QPushButton {border-image: url(assets/play.png); background-color: transparent}}")
            self.player.pause()
            # self.player.stop()

    def updateSlider(self):
        # update slider when playing
        if self.playFlag:
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.player.duration())
            self.slider.setValue(self.slider.value() + 1000) # update

        self.startTimeLabel.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.endTimeLabel.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        # reset the slider when playing over
        if self.player.position() == self.player.duration():
            self.slider.setValue(0)
            self.startTimeLabel.setText('00:00')
            self.playFlag = False
            self.player.stop()
            # self.playButton.setText('play')
            self.playButton.setStyleSheet("QPushButton {border-image: url(assets/play.png); background-color: transparent}}")
            self.convertButton.setDisabled(False)
            self.convertButton.setStyleSheet("QPushButton {border-image: url(assets/start.png); background-color: transparent}}")


    # TODO: 调节音量
    def changeVolume(self):
        pass

    def downloadAudio(self):
        file_path = QFileDialog.getSaveFileName(self, 'Chosse save file path', 'converted.wav', f'(*.wav)')
        if file_path[0]:
            try:
                shutil.copy(self.download_path, file_path[0])
            except:
                print('Nothing to download!')

    # hubert or flow
    def changeMode(self):
        # hubert -> flow
        if not self.mode:
            self.mode = True
            self.modeButton.setStyleSheet("QPushButton {border-image: url(assets/mode-flow.png); background-color: transparent}")
        # flow -> hubert
        else:
            self.mode = False
            self.modeButton.setStyleSheet("QPushButton {border-image: url(assets/mode-hubert.png); background-color: transparent}")

        StellaVC.change_mode()


    def closeEvent(self, e):
        reply = QMessageBox.question(self,
                                     'Question',
                                     "Are you sure to close StellaVC?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            e.accept()
            sys.exit(0)  # 退出程序
        else:
            e.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BaseGuiWidget()
    print('Welcome to Stella Voice Changer!\n\n'
          'Please specify the model and configuration file paths to load models!')
    sys.exit(app.exec_())
