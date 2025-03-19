#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates

# If you just want to see positions for one specific model, put its name here:
TARGET_MODEL = "quadrotor"  # or "vessel", etc. If you want them all, skip filtering.

def callback(msg):
    # msg.name is a list of model names, e.g. ["ocean_waves", "vessel", ...]
    # msg.pose is a corresponding list of geometry_msgs/Pose
    # msg.twist is a corresponding list of geometry_msgs/Twist
    
    for i, model_name in enumerate(msg.name):
        if TARGET_MODEL and model_name != TARGET_MODEL:
            continue  # Skip other models
        
        pos = msg.pose[i].position
        # Print or log the position data
        rospy.loginfo("Model: %s, Position: x=%.3f y=%.3f z=%.3f", 
                      model_name, pos.x, pos.y, pos.z)
        # Optionally, write to a file (CSV):
        # with open("/home/user/positions.csv", "a") as f:
        #     f.write(f"{rospy.Time.now().to_sec()},{model_name},{pos.x},{pos.y},{pos.z}\n")

def main():
    rospy.init_node("extract_pose_node")
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
    rospy.spin()

if __name__ == "__main__":
    main()
