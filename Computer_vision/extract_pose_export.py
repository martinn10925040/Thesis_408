#!/usr/bin/env python2
import rospy
from gazebo_msgs.msg import ModelStates

TARGET_MODEL = "quadrotor"

# Change this path to wherever you want the CSV to be saved
OUTPUT_FILE = "/mnt/c/Users/Zephyrus GA401/Desktop/UNI-LIFE/Thesis-408/ROS/subset_20250312/pose_data.csv"

def callback(msg):
    """
    Callback for /gazebo/model_states. We look for the model named TARGET_MODEL,
    print its position to the console, and append it to a CSV file.
    """
    for i, model_name in enumerate(msg.name):
        if TARGET_MODEL and model_name != TARGET_MODEL:
            continue

        pos = msg.pose[i].position
        t = rospy.Time.now().to_sec()

        # Print to console
        rospy.loginfo("Time: %.3f | Model: %s | X: %.3f | Y: %.3f | Z: %.3f",
                      t, model_name, pos.x, pos.y, pos.z)

        # Append to CSV file
        with open(OUTPUT_FILE, "a") as f:
            f.write("{},{},{},{},{}\n".format(t, model_name, pos.x, pos.y, pos.z))

def on_shutdown():
    """
    This function is called automatically when the node is shutting down.
    """
    rospy.loginfo("Finished writing data to %s", OUTPUT_FILE)

def main():
    rospy.init_node("extract_pose_node", anonymous=True)
    
    # Overwrite or create a fresh file with a header
    with open(OUTPUT_FILE, "w") as f:
        f.write("time,model_name,x,y,z\n")

    # Subscribe to /gazebo/model_states
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)

    # Register a shutdown hook to print a final message
    rospy.on_shutdown(on_shutdown)

    rospy.loginfo("Node started. Subscribing to /gazebo/model_states.")
    rospy.loginfo("Writing data to %s", OUTPUT_FILE)

    # Keep the node alive until shutdown (e.g. bag finishes or Ctrl+C)
    rospy.spin()

if __name__ == "__main__":
    main()
