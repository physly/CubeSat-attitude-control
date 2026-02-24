import rowan
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Your functions
def quaternion_from_bases(bases):
    r11, r12, r13 = bases[0]
    r21, r22, r23 = bases[1]
    r31, r32, r33 = bases[2]

    q0 = 1.0 + r11 + r22 + r33
    q1 = 1.0 + r11 - r22 - r33
    q2 = 1.0 - r11 + r22 - r33
    q3 = 1.0 - r11 - r22 + r33
    q_max = max(q0, q1, q2, q3)

    if q_max == q0:
        w = 0.5 * np.sqrt(q0)
        x = (r32 - r23) / (4.0 * w)
        y = (r13 - r31) / (4.0 * w)
        z = (r21 - r12) / (4.0 * w)
    elif q_max == q1:
        x = 0.5 * np.sqrt(q1)
        w = (r32 - r23) / (4.0 * x)
        y = (r12 + r21) / (4.0 * x)
        z = (r13 + r31) / (4.0 * x)
    elif q_max == q2:
        y = 0.5 * np.sqrt(q2)
        w = (r13 - r31) / (4.0 * y)
        x = (r12 + r21) / (4.0 * y)
        z = (r23 + r32) / (4.0 * y)
    else:
        z = 0.5 * np.sqrt(q3)
        w = (r21 - r12) / (4.0 * z)
        x = (r13 + r31) / (4.0 * z)
        y = (r23 + r32) / (4.0 * z)

    quatern = np.array([w, x, y, z])
    return quatern / np.linalg.norm(quatern)

def compute_tangents(quaternions):
    """Calculate tangents for SQUAD interpolation"""
    n = len(quaternions)
    tangents = np.zeros_like(quaternions)
    
    if n <= 2:
        return quaternions
    
    for i in range(1, n-1):
        q_prev = quaternions[i-1]
        q_curr = quaternions[i]
        q_next = quaternions[i+1]
        
        # Ensure quaternions are normalized
        q_curr = q_curr / np.linalg.norm(q_curr)
        
        # Calculate logarithms
        inv_q = rowan.conjugate(q_curr)
        rel_q1 = rowan.multiply(inv_q, q_prev)
        rel_q2 = rowan.multiply(inv_q, q_next)
        
        log1 = rowan.log(rel_q1)
        log2 = rowan.log(rel_q2)
        
        # Average value
        avg_log = -0.25 * (log1 + log2)
        
        # Exponential and multiplication
        tangents[i] = rowan.multiply(q_curr, rowan.exp(avg_log))
    
    # Boundary cases
    tangents[0] = quaternions[0]
    tangents[-1] = quaternions[-1]
    
    return tangents

def get_interpolated_quaternion(key_times, key_quaternions, current_time):
    """Get interpolated quaternion for current time"""
    # Normalize all quaternions
    key_quaternions = np.array([q / np.linalg.norm(q) for q in key_quaternions])
    
    # If time is before first key
    if current_time <= key_times[0]:
        return key_quaternions[0]
    
    # If time is after last key
    if current_time >= key_times[-1]:
        return key_quaternions[-1]
    
    # Find interval
    idx = np.searchsorted(key_times, current_time) - 1
    
    # Calculate tangents
    tangents = compute_tangents(key_quaternions)
    
    # Normalize time to [0, 1] interval
    t_norm = (current_time - key_times[idx]) / (key_times[idx+1] - key_times[idx])
    
    # Ensure t_norm is in [0, 1]
    t_norm = max(0.0, min(1.0, t_norm))
    
    # Get quaternions for interpolation
    q0 = key_quaternions[idx]
    q1 = key_quaternions[idx+1]
    s0 = tangents[idx]
    s1 = tangents[idx+1]
    
    # Check quaternion shapes
    if q0.ndim == 2 and q0.shape[0] == 1:
        q0 = q0[0]
    if q1.ndim == 2 and q1.shape[0] == 1:
        q1 = q1[0]
    if s0.ndim == 2 and s0.shape[0] == 1:
        s0 = s0[0]
    if s1.ndim == 2 and s1.shape[0] == 1:
        s1 = s1[0]
    
    # SLERP or SQUAD interpolation
    if np.array_equal(s0, q0) and np.array_equal(s1, q1):
        # If tangents equal quaternions, use SLERP
        q_interp = rowan.interpolate.slerp(q0, q1, t_norm)
    else:
        # Use SQUAD
        # Create arrays with correct shape
        q0 = np.array(q0).flatten()
        q1 = np.array(q1).flatten()
        s0 = np.array(s0).flatten()
        s1 = np.array(s1).flatten()
        
        # SQUAD interpolation
        q_interp = rowan.interpolate.squad(q0, s0, s1, q1, t_norm)
    
    # Normalize result
    q_interp = np.array(q_interp).flatten()
    return q_interp / np.linalg.norm(q_interp)

def quaternion_to_targetx(q):
    """Convert quaternion to targetx (0-320)"""
    # Ensure q is a flat array with 4 elements
    q = np.array(q).flatten()
    
    if len(q) != 4:
        print(f"Error: quaternion must have 4 elements, got {len(q)}")
        return 160
    
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Get rotation angle around Z axis (yaw) from quaternion
    # q = [w, x, y, z]
    
    # For rotation around Z axis
    # Formula to extract yaw from quaternion
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Limit angle to ±90 degrees (±π/2 radians)
    yaw = np.clip(yaw, -np.pi/2, np.pi/2)
    
    # Convert to targetx: -π/2..π/2 → 0..320
    targetx = (yaw + np.pi/2) * (320 / np.pi)
    
    # Round to integer
    return int(np.clip(targetx, 0, 320))

def targetx_to_quaternion(targetx):
    """Convert targetx to quaternion"""
    # targetx: 0..320 → angle: -π/2..π/2
    angle = (targetx - 160) * (np.pi / 2) / 160.0
    
    # Quaternion for rotation around Z axis
    q = np.array([
        np.cos(angle/2),  # w
        0,                # x  
        0,                # y
        np.sin(angle/2)   # z
    ])
    
    return q / np.linalg.norm(q)

def create_test_sequence_with_pause():
    """Create test sequence with 3-second pause at 160"""
    
    # Time marks (seconds from start)
    # Add extra points to create pause
    key_times = np.array([0.0, 2.0, 5.0, 7.0, 10.0])
    
    # Target values with pause at 160
    # Scheme: 80 → 160 (2 sec) → pause 3 sec → 80 (2 sec) → 160 (3 sec)
    target_values = [80, 160, 160, 80, 160]
    
    # Convert targetx to quaternions
    key_quaternions = []
    
    for targetx in target_values:
        q = targetx_to_quaternion(targetx)
        key_quaternions.append(q)
    
    key_quaternions = np.array(key_quaternions)
    
    return key_times, key_quaternions, target_values

def send_command(command):
    """Command sending function (simulation)"""
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {command}")

def run_simulation_with_pause(send_interval=0.01):
    """Run real-time simulation with pause"""
    
    print("=" * 70)
    print("SMOOTH TARGETX INTERPOLATION SIMULATION WITH 3-SECOND PAUSE AT 160")
    print("=" * 70)
    
    # Create test sequence with pause
    key_times, key_quaternions, target_values = create_test_sequence_with_pause()
    
    print("\nSEQUENCE WITH PAUSE:")
    print("(time is measured from start)")
    print("-" * 70)
    
    for i, (t, tx) in enumerate(zip(key_times, target_values)):
        if i < len(key_times) - 1:
            duration = key_times[i+1] - t
            if tx == target_values[i+1]:
                print(f"  t={t:4.1f}s: targetx={tx:3d}  → HOLD for {duration:4.1f} seconds")
            else:
                print(f"  t={t:4.1f}s: targetx={tx:3d}  → TRANSITION to {target_values[i+1]:3d} in {duration:4.1f} seconds")
        else:
            print(f"  t={t:4.1f}s: targetx={tx:3d}  → FINAL POSITION")
    
    print(f"\nTotal execution time: {key_times[-1]:.1f} seconds")
    print(f"Update frequency: {1/send_interval:.0f} Hz ({send_interval*1000:.0f} ms)")
    print("-" * 70)
    
    # Start timer
    start_time = time.time()
    end_time = key_times[-1]
    
    # History for plotting
    history_time = []
    history_targetx = []
    
    # Main loop
    print("\nSTARTING COMMAND SENDING...\n")
    
    try:
        while True:
            # Current time from start
            current_time = time.time() - start_time
            
            # Stop if we passed all key points + small delay
            if current_time > end_time + 1.0:  # +1 second for completion
                print(f"\nSimulation completed (total time: {current_time:.1f}s)")
                break
            
            # Get interpolated quaternion
            q_interp = get_interpolated_quaternion(key_times, key_quaternions, current_time)
            
            # Convert to targetx
            targetx = quaternion_to_targetx(q_interp)
            
            # Save history
            history_time.append(current_time)
            history_targetx.append(targetx)
            
            # Send command
            command = f"PID:T{targetx}"
            send_command(command)
            
            # Wait for next update
            time.sleep(send_interval)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    return np.array(history_time), np.array(history_targetx), key_times, target_values

def plot_results_with_pause(history_time, history_targetx, key_times, target_values):
    """Plot results with pause"""
    plt.figure(figsize=(14, 10))
    
    # 1. Interpolation plot
    plt.subplot(3, 1, 1)
    
    # Main interpolation line
    plt.plot(history_time, history_targetx, 'b-', alpha=0.7, linewidth=2, label='SQUAD interpolation')
    
    # Key points
    plt.scatter(key_times, target_values, color='red', s=100, zorder=5, label='Key points')
    
    # Pause regions
    for i in range(len(key_times)-1):
        if target_values[i] == target_values[i+1]:
            # This is a pause
            plt.axvspan(key_times[i], key_times[i+1], alpha=0.1, color='green', label='Pause' if i==2 else "")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('targetx')
    plt.title('Smooth targetx interpolation with 3-second pause at 160')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(70, 170)
    
    # Add annotations
    plt.annotate('3-second pause', xy=(3.5, 165), xytext=(2, 175),
                 arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10)
    
    # 2. Velocity plot
    plt.subplot(3, 1, 2)
    if len(history_time) > 1:
        # Calculate velocity
        dt = np.diff(history_time)
        dtargetx = np.diff(history_targetx)
        velocity = dtargetx / dt
        
        plt.plot(history_time[:-1], velocity, 'g-', alpha=0.7, linewidth=2)
        plt.fill_between(history_time[:-1], velocity, alpha=0.3, color='green')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (targetx/second)')
        plt.title('Targetx change velocity (zero velocity = pause)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Highlight pause regions
        for i in range(len(key_times)-1):
            if target_values[i] == target_values[i+1]:
                plt.axvspan(key_times[i], key_times[i+1], alpha=0.1, color='gray')
    
    # 3. Acceleration plot
    plt.subplot(3, 1, 3)
    if len(velocity) > 1:
        acceleration = np.diff(velocity) / dt[:-1]
        plt.plot(history_time[:-2], acceleration, 'r-', alpha=0.7, linewidth=2)
        plt.fill_between(history_time[:-2], acceleration, alpha=0.3, color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Acceleration (targetx/second²)')
        plt.title('Targetx change acceleration (smooth transitions)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='b', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 50)
    print("EXECUTION STATISTICS:")
    print("=" * 50)
    print(f"Total simulation time: {history_time[-1]:.2f} s")
    print(f"Number of commands sent: {len(history_time)}")
    print(f"Average targetx: {np.mean(history_targetx):.1f}")
    print(f"Maximum velocity: {np.max(np.abs(velocity)):.1f} targetx/second")
    if 'acceleration' in locals():
        print(f"Maximum acceleration: {np.max(np.abs(acceleration)):.1f} targetx/second²")
    
    # Analyze pauses
    print("\nPAUSE ANALYSIS:")
    for i in range(len(key_times)-1):
        if target_values[i] == target_values[i+1]:
            start_idx = np.argmax(history_time >= key_times[i])
            end_idx = np.argmax(history_time >= key_times[i+1])
            if end_idx == 0:  # if not found, take last index
                end_idx = len(history_time) - 1
            
            pause_duration = key_times[i+1] - key_times[i]
            pause_target = target_values[i]
            
            # Check stability during pause
            if end_idx > start_idx:
                pause_values = history_targetx[start_idx:end_idx]
                std_dev = np.std(pause_values)
                print(f"  Pause at targetx={pause_target}:")
                print(f"    Duration: {pause_duration:.1f} s")
                print(f"    Stability (σ): {std_dev:.2f}")

def main_with_pause():
    """Main function with pause"""
    # Run simulation with pause
    history_time, history_targetx, key_times, target_values = run_simulation_with_pause(send_interval=0.01)
    
    # Plot if we have data
    if len(history_time) > 0:
        print("\nGenerating plot with pause...")
        plot_results_with_pause(history_time, history_targetx, key_times, target_values)
    
    print("\nPause simulation completed successfully!")

if __name__ == "__main__":
    main_with_pause()
