"""
Simplified debug script to understand RS007L joint geometry.
"""
import numpy as np
import one.utils.math as oum
from one.robots.manipulators.kawasaki.rs007l import rs007l

def main():
    robot = rs007l.RS007L()
    
    print("=" * 80)
    print("RS007L Joint Geometry at Zero Configuration")
    print("=" * 80)
    
    # Get joints from the chain
    jnts = robot._chain.jnts
    
    for i in range(6):
        jnt = jnts[i]
        pos = jnt.pos
        axis = jnt.ax
        print(f"\nJ{i+1}:")
        print(f"  Position: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        print(f"  Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    
    # Calculate key distances
    o1 = jnts[0].pos
    o2 = jnts[1].pos
    o3 = jnts[2].pos
    o4 = jnts[3].pos
    
    print("\n" + "=" * 80)
    print("Key Distances:")
    print("=" * 80)
    print(f"||o2 - o1|| = {np.linalg.norm(o2 - o1):.6f} m")
    print(f"||o3 - o2|| = {np.linalg.norm(o3 - o2):.6f} m (l2, shoulder to elbow)")
    print(f"||o4 - o3|| = {np.linalg.norm(o4 - o3):.6f} m (l3, elbow to wrist)")
    
    # Test FK with q1=0, q2=90, q3=0
    print("\n" + "=" * 80)
    print("Test Configuration: q1=0°, q2=90°, q3=0°")
    print("=" * 80)
    qs = np.array([0, np.pi/2, 0, 0, 0, 0], dtype=np.float32)
    robot.fk(qs)
    
    print("\nJoint positions after FK:")
    for i in range(4):
        pos = jnts[i].pos
        print(f"  J{i+1}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    
    wrist_pos = jnts[3].pos
    print(f"\nWrist center (J4): [{wrist_pos[0]:.6f}, {wrist_pos[1]:.6f}, {wrist_pos[2]:.6f}]")
    
    # Test FK with q1=90, q2=0, q3=0  
    print("\n" + "=" * 80)
    print("Test Configuration: q1=90°, q2=0°, q3=0°")
    print("=" * 80)
    qs = np.array([np.pi/2, 0, 0, 0, 0, 0], dtype=np.float32)
    robot.fk(qs)
    
    print("\nJoint positions after FK:")
    for i in range(4):
        pos = jnts[i].pos
        print(f"  J{i+1}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    
    wrist_pos = jnts[3].pos
    print(f"\nWrist center (J4): [{wrist_pos[0]:.6f}, {wrist_pos[1]:.6f}, {wrist_pos[2]:.6f}]")
    
    # Test FK with q1=90, q2=90, q3=180
    print("\n" + "=" * 80)
    print("Test Configuration: q1=90°, q2=90°, q3=180°")
    print("=" * 80)
    qs = np.array([np.pi/2, np.pi/2, np.pi, 0, 0, 0], dtype=np.float32)
    robot.fk(qs)
    
    print("\nJoint positions after FK:")
    for i in range(4):
        pos = jnts[i].pos
        print(f"  J{i+1}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    
    wrist_pos = jnts[3].pos
    print(f"\nWrist center (J4): [{wrist_pos[0]:.6f}, {wrist_pos[1]:.6f}, {wrist_pos[2]:.6f}]")
    print(f"Target was: [0.283, 0, 0.643]")
    error = np.linalg.norm(wrist_pos - np.array([0.283, 0, 0.643]))
    print(f"Error: {error:.6f} m")

if __name__ == '__main__':
    main()
