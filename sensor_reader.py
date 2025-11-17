import serial
import time

def get_sensor_data(port='COM9', baudrate=9600, timeout=2):
    """
    Reads humidity, temperature, and soil moisture from Arduino.
    Returns:
        dict: {'humidity': 65.0, 'temperature': 28.7, 'soil_moisture': 45.0}
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(3)  # wait for Arduino to reset
        ser.flushInput()

        print("🔌 Reading live sensor data from Arduino...")

        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                print("⚠️ Timeout: No valid data received from Arduino.")
                ser.close()
                return None

            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if not raw:
                continue

            parts = raw.split(',')
            if len(parts) != 3:
                continue

            try:
                h, t, s = map(float, parts)
                ser.close()
                print(f"✅ Data → Humidity={h:.2f}% | Temp={t:.2f}°C | Soil={s:.2f}%")
                return {'humidity': h, 'temperature': t, 'soil_moisture': s}
            except ValueError:
                continue

    except serial.SerialException as e:
        print(f"❌ SERIAL ERROR: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None
