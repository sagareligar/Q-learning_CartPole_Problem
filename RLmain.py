#!/usr/bin/env python
import random
import math
import sys
import time 
from Kalman import KalmanAngle

from reinforcementLearner import *



kalmanX = KalmanAngle()
kalmanY = KalmanAngle()

RestrictPitch = False	
radToDeg = 57.2957786
kalAngleX = 0
kalAngleY = 0
#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47


# Python file to launch simulation and handling vrep

debug = 0

temp1=1





#Read the gyro and acceleromater values from MPU6050
def MPU_Init():
	#write to sample rate register
	bus.write_byte_data(DeviceAddress, SMPLRT_DIV, 7)

	#Write to power management register
	bus.write_byte_data(DeviceAddress, PWR_MGMT_1, 1)

	#Write to Configuration register
	#Setting DLPF (last three bit of 0X1A to 6 i.e '110' It removes the noise due to vibration.) https://ulrichbuschbaum.wordpress.com/2015/01/18/using-the-mpu6050s-dlpf/
	bus.write_byte_data(DeviceAddress, CONFIG, int('0000110',2))

	#Write to Gyro configuration register
	bus.write_byte_data(DeviceAddress, GYRO_CONFIG, 24)

	#Write to interrupt enable register
	bus.write_byte_data(DeviceAddress, INT_ENABLE, 1)


def read_raw_data(addr):
	#Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(DeviceAddress, addr)
        low = bus.read_byte_data(DeviceAddress, addr+1)

        #concatenate higher and lower value
        value = ((high << 8) | low)

        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value

def read_variables_():

	accX = read_raw_data(ACCEL_XOUT_H)
	accY = read_raw_data(ACCEL_YOUT_H)
	accZ = read_raw_data(ACCEL_ZOUT_H)

	    #Read Gyroscope raw value
	gyroX = read_raw_data(GYRO_XOUT_H)
	gyroY = read_raw_data(GYRO_YOUT_H)
	gyroZ = read_raw_data(GYRO_ZOUT_H)

	dt = time.time() - timer
	timer = time.time()
    
 	roll = math.atan(accY/math.sqrt((accX**2)+(accZ**2))) * radToDeg
	pitch = math.atan2(-accX,accZ) * radToDeg
	gyroXRate = gyroX/131
	gyroYRate = gyroY/131

	if((pitch < -90 and kalAngleY >90) or (pitch > 90 and kalAngleY < -90)):

	    kalmanY.setAngle(pitch)
	    complAngleY = pitch
	    kalAngleY   = pitch
	    gyroYAngle  = pitch
	else:
		kalAngleY = kalmanY.getAngle(pitch,gyroYRate,dt)


	if(abs(kalAngleY)>90):

 		gyroXRate  = -gyroXRate
    	kalAngleX = kalmanX.getAngle(roll,gyroXRate,dt)

		#angle = (rate of change of angle) * change in time
	gyroXAngle = gyroXRate * dt
	gyroYAngle = gyroYAngle * dt

		#compAngle = constant * (old_compAngle + angle_obtained_from_gyro) + constant * angle_obtained from accelerometer
	compAngleX = 0.93 * (compAngleX + gyroXRate * dt) + 0.07 * roll
	compAngleY = 0.93 * (compAngleY + gyroYRate * dt) + 0.07 * pitch

	if ((gyroXAngle < -180) or (gyroXAngle > 180)):
	    gyroXAngle = kalAngleX
	if ((gyroYAngle < -180) or (gyroYAngle > 180)):
	    gyroYAngle = kalAngleY

	return kalmanY



if __name__ == '__main__':
	bus = smbus.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
	DeviceAddress = 0x68   # MPU6050 device address

	MPU_Init()

	time.sleep(1)
#Read Accelerometer raw value
	accX = read_raw_data(ACCEL_XOUT_H)
	accY = read_raw_data(ACCEL_YOUT_H)
	accZ = read_raw_data(ACCEL_ZOUT_H)



	roll = math.atan(accY/math.sqrt((accX**2)+(accZ**2))) * radToDeg
	pitch = math.atan2(-accX,accZ) * radToDeg
	# print(roll)
	kalmanX.setAngle(roll)
	kalmanY.setAngle(pitch)
	gyroXAngle = roll;
	gyroYAngle = pitch;
	compAngleX = roll;
	compAngleY = pitch;

	
	flag = 0






       


        # create a new reinforcement learner
        cart = ReinforcementLearner()

        # initialize values
        p, oldp, rhat, r = 0, 0, 0, 0
        state, i, y, steps, failures, failed, startSim = 0, 0, 0, 0, 0, False, True

        while steps < cart.max_steps and failures < cart.max_failures:
            # start simulation in the first step
            if startSim == True:


                
                now = int(time.time())
                timer = time.time()
                # get start state
                cart.t=read_variables_()

                state = cart.get_state()


            # guess new action depending on the weight of the current state
            action = (random.random()/((2**31) - 1) < (1.0 / (1.0 + math.exp(-max(-50, min(cart.action_weights[state], 50))))))
            if debug == 1:
                print "action: " + str(action)

            #update traces
            cart.action_weights_elig[state] += (1 - cart.lambda_w) * (y - 0.5)
            cart.critic_weights_elig[state] += (1 - cart.lambda_v)
            oldp = cart.critic_weights[state]     # remember prediction for the current state
            cart.do_action(action)                # do action
            cart.t=read_variables_()
              # read new values TODO maybe a bit to close after doing action?!
            state = cart.get_state()              # get new x, dx, t, dt


            # failure
            if state < 0:
                failed = True
                failures += 1
                print "Trial " + str(failures) + " was " + str(steps) + " steps or " + str(int(time()) - now) + " seconds"        
                steps = 0
                # restart simulation and get initial start values


                
                #DO the PID here




                time.sleep(0.5)      
                state = cart.get_state()

                cart.t=read_variables_()

                       
                r = -1  # reward = -1
                p = 0   # prediction of failure
                startSim = True
            else:     # no failure
                failed = False
                r = 0   # reward = 0
                p = cart.critic_weights[state]

            rhat = r + cart.gamma * p - oldp

            # update all weights
            cart.update_all_weights(rhat, failed)

            # going to next step
            steps += 1

        # finished the loop
        if failures == cart.max_failures:
            print "Pole not balanced. Stopping after " + str(failures) + " failures \n"
        else:
            print "Pole balanced successfully for at least " + str(steps) + " steps \n"
        print "critic weights: " + str(cart.critic_weights)
        print "action weights: " + str(cart.action_weights)