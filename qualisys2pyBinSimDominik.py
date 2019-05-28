"""
This script will receive the 3 rotation angles Pitch/Yaw/Roll around the 3 Axes X/Y/Z
from the Qualisys Track Manager and send it to pyBinSim.
???It currently only works with Windows 7+.??

@original author: Stephan Fremerey @pybinsim adaptation: Florian Klein
"""

from OSC import *
import qtm
import math
import msvcrt
import csv
from datetime import datetime

ip = "127.0.0.1"  # set to IP address of target computer
port = 10000  # pyBinSim Port
oscIdentifier = '/pyBinSim'




class QScript:
    """
    Class containing functions to get access to PYR data of Qualisys system.
    See https://github.com/qualisys/qualisys_python_sdk/blob/master/examples/basic_example.py
    """

    def __init__(self):
        self.qrt = qtm.QRT("127.0.0.1", 22223)  # Qualysis Port
        self.qrt.connect(on_connect=self.on_connect, on_disconnect=self.on_disconnect, on_event=self.on_event)

        # Vectors to enter available filter data
        self.nChannels = 6
        self.yawVectorAvailableData = range(0, 360, 5)
        self.xPositionAvailableData = range(1,17)
        self.YPositionAvailableData = range(1,17)
        self.xPositionOffset = -225
        self.yPositionOffset = -225
        self.yawOffset = 0

        # one step relates to 25cm
        self.stepSize = 25

        # Create OSC client
        self.client = OSCMultiClient()
        self.client.setOSCTarget((ip, port))

        # Defaults
        self.posX, self.posY, self.posZ, self.roll, self.pitch, self.yaw = [0, 0, 0, 0, 0, 0]

    def on_connect(self, connection, version):
        print('Connected to QTM with {}'.format(version))
        # Connection is the object containing all methods/commands you can send to qtm
        self.connection = connection

        # Start RT-Stream live
        self.start_stream()

    def on_disconnect(self, reason):
        print(reason)
        # Stops main loop and exits script
        qtm.stop()

    def on_event(self, event):
        # Print event type
        print(event)

    def on_error(self, error):
        error_message = error.getErrorMessage()
        if error_message == "'RT from file already running'":
            # If rt already is running we can start the stream anyway
            self.start_stream()
        else:
            # On other errors we fail
            print(error_message)
            self.connection.disconnect()

    def on_packet(self, packet):
        # Ask for keyboard interaction
        if msvcrt.kbhit():
            char = msvcrt.getch()

        # Get the Pitch, Yaw and Roll data out of the sent package and send it to a UDP socket.
        header, sixd_euler_data = packet.get_6d_euler()

        posX, posY, posZ, roll, pitch, yaw = [0, 0, 0, 0, 0, 0]

        for body_index, body in enumerate(sixd_euler_data, 1):
            if body_index == 1:
                posX, posY, posZ = body[0]
                roll, pitch, yaw = body[1]
            else:
                print('Only data of body 1 is processed')

        if not math.isnan(yaw):
            self.yaw = yaw
            # Transform yaw: 0 to 360
            self.yaw = int(round(self.yaw)) - self.yawOffset
            if self.yaw < 0:
                self.yaw += 360

        if not math.isnan(pitch):
            self.pitch = pitch
            self.pitch = int(round(self.pitch))
            if self.pitch < 0:
                self.pitch += 360
				
        if not math.isnan(roll):
            self.roll = roll
            self.roll = int(round(self.roll))
            if self.roll < 0:
                self.roll += 360
			
        if not math.isnan(posX):
            self.posX = -posX
            #self.posX = -posX
            # Transform posX to cm
            self.posX = (round(self.posX) / 10) - self.xPositionOffset

            # Transform from cm to self.xPositionAvailableData
            self.posX = self.posX/self.stepSize

        if not math.isnan(posY):
            self.posY = posY
            # Transform posX to cm
            self.posY = (round(self.posY) / 10) - self.yPositionOffset

            # Transform from cm to self.xPositionAvailableData
            self.posY = self.posY/self.stepSize

        if not math.isnan(posZ):
            self.posZ = posZ
            # Transform posX to cm
            self.posZ = (round(self.posZ) / 10)

        #print(self.posX, self.posY, self.posZ, self.roll, self.pitch, self.yaw)
        dt = datetime.now()
        data_list = [dt.strftime("%d%m%Y-%H:%M:%S:%f"), self.posX, self.posY, self.posZ, self.roll, self.pitch, self.yaw]
        print(data_list)
        with open("test.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data_list)
			
        # print("X Position: %s, Y Position: %s , Z Position: %s" % (self.posX, self.posY, self.posZ))
        # print("Yaw: %s " % self.yaw)

        # Select nearest values according to available filters
        nearest_yaw = min(self.yawVectorAvailableData, key=lambda x: abs(x - self.yaw))
        nearest_posX = min(self.xPositionAvailableData, key=lambda x: abs(x - self.posX))
        nearest_posY = min(self.YPositionAvailableData, key=lambda x: abs(x - self.posY))
		
        
        # Send to pyBinSim
        for i in range(1, self.nChannels+1):
            sourceChannel = outputChannel = i
            binSimParameters = [sourceChannel-1, 0, nearest_posY, nearest_posX, outputChannel, str(nearest_yaw).zfill(3), 0]

            #print('Debug: ', binSimParameters)
            message = OSCMessage(oscIdentifier)
            message.append(binSimParameters)
            self.client.send(message)

    def start_stream(self):
        # Start streaming 6d data and register packet callback
        self.connection.stream_frames(on_packet=self.on_packet, components=['6deuler'])

        # Schedule a call for later to shutdown connection
        # qtm.call_later(5, self.connection.disconnect)


def main():
    # Instantiate our script class
    # We don't need to create a class, you could also store the connection object in a global variable
    QScript()

    # Start the processing loop
    print("Sending data to pyBinSim now...")
    qtm.start()


if __name__ == '__main__':
    main()
