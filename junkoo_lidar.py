import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import rospy
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

pcd = o3d.io.read_point_cloud("/Users/jm/git/template_matching_multiple_ROI/resrc/lidar_sample.ply")

def pcd_callback(data):
    cloud = pcl_helper.ros_to_pcl(data)

def main():
    
    rospy.init_node('lidar', anonymous=True)
    pub = rospy.Publisher('ROI_data', Int32MultiArray, queue_size=100)
    rospy.Subscriber('/velodyne_points', PointCloud2, pcd_callback)
    rate = rospy.Rate(5) # 10hz

    while not rospy.is_shutdown():
        picked_pcd = o3d.geometry.PointCloud()
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=50, min_points=50, print_progress=False))
        max_label = labels.max()
        # print(f"point cloud has {max_label} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # picked_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points, dtype=np.float32)[labels == most_frequent(list(labels))])
        bouning_yz = []
        bouning_wh = []
        bounding_points = []
        for label in range(max_label+1):
            bounding_points.append(np.asarray(pcd.points, dtype=np.float32)[labels == label])
            picked_pcd = o3d.geometry.PointCloud()
            picked_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points, dtype=np.float32)[labels == label])
            aabb = picked_pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)

            x_axis = np.array([1,0,0])
            dist = calc_projection(aabb.get_center(), x_axis)
            projected_point = aabb.get_center() - x_axis * dist
            bouning_yz.append(projected_point[1:3])
            bouning_wh.append(aabb.get_extent()[1:3])
            
            # temp=[]
            # x_axis = np.array([1,0,0])
            # for pnt in aabb.get_box_points():
            #     dist = calc_projection(pnt, x_axis)
            #     projected_point = pnt - x_axis * dist
            #     temp.append(projected_point)
            # projected_box = o3d.geometry.LineSet()
            # projected_box = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(temp), lines=o3d.utility.Vector2iVector(lines))
            # # o3d.visualization.draw_geometries([pcd, aabb, projected_box])
        
            o3d.visualization.draw_geometries([pcd, aabb])
            print(bouning_yz)
            print(bouning_wh)

            pub_data = np.array(bouning_yz + bouning_wh)
            pub.publish(Int32MultiArray(data=pub_data))
            rate.sleep()
        rospy.spin()


def calc_projection(a, b):
    # return (np.linalg.norm((np.dot(a, b) / np.dot(b, b)) * b))  # 기존 projection
    return np.dot(a, b) / np.linalg.norm(b)  # 개선된 projection
lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

def most_frequent(data):
    return max(data, key=data.count)

if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    main()
    vis.destroy_window()