from flask import Flask, request, render_template_string
from adafruit_servokit import ServoKit
import threading
import time
import lgpio  # Motor control with PWM

# --- Initialize Hardware ---
kit = ServoKit(channels=16)

IN1 = 22
IN2 = 23
ENA = 27
IN3 = 5
IN4 = 6
ENB = 13

# Speed control state
current_speed = 100  # 0–100%

try:
    h = lgpio.gpiochip_open(0)
    for pin in [IN1, IN2, ENA, IN3, IN4, ENB]:
        lgpio.gpio_claim_output(h, pin, 0)

    def set_motor_speed(duty_percent):
        global current_speed
        current_speed = duty_percent
        lgpio.tx_pwm(h, ENA, 1000, duty_percent)
        lgpio.tx_pwm(h, ENB, 1000, duty_percent)
        print(f"Speed set to {duty_percent}%")

    set_motor_speed(current_speed)
    print("Motor GPIO initialized successfully.")
except Exception as e:
    print(f"Error initializing lgpio: {e}")
    h = None

app = Flask(__name__)

# --- Servo Movement Functions ---

def set_servo_angles(angle_dict):
    for ch, angle in angle_dict.items():
        if 0 <= ch < 16:
            kit.servo[ch].angle = angle

def hiding(): print("Executing: HIDING")

def nod_no():
    def routine():
        print("Executing: NOD_NO")
        try:
            original_angle = kit.servo[2].angle
            kit.servo[2].angle = 50
            time.sleep(0.3)
            kit.servo[2].angle = 130
            time.sleep(0.3)
            kit.servo[2].angle = 50
            time.sleep(0.3)
            kit.servo[2].angle = 130
            time.sleep(0.3)
            kit.servo[2].angle = original_angle
        except Exception as e:
            print(f"Error in nod_no: {e}")
    threading.Thread(target=routine).start()

def nod_yes():
    def routine():
        print("Executing: NOD_YES")
        try:
            original_angle = kit.servo[3].angle
            kit.servo[3].angle = 130
            time.sleep(0.3)
            kit.servo[3].angle = 50
            time.sleep(0.3)
            kit.servo[3].angle = 130
            time.sleep(0.3)
            kit.servo[3].angle = 50
            time.sleep(0.3)
            kit.servo[3].angle = original_angle
        except Exception as e:
            print(f"Error in nod_yes: {e}")
    threading.Thread(target=routine).start()

def dance():
    def routine():
        print("Executing: DANCE")
        steps = [
            {0: 40, 1: 140, 2: 90, 3: 180, 4: 90, 5: 180, 6: 0},
            {3: 90, 4: 180, 5: 90, 6: 90},
            {3: 180, 4: 90, 5: 90, 6: 90},
            {3: 90, 4: 180, 5: 180, 6: 0},
            {3: 180, 4: 90, 5: 90, 6: 90},
            {3: 90, 4: 180, 5: 90, 6: 90}
        ]
        for step in steps:
            set_servo_angles(step)
            time.sleep(0.3)
        reset_servos()
    threading.Thread(target=routine).start()

def reset_back_servos(): set_servo_angles({0: 40, 1: 120, 2: 90, 3: 90, 4: 90, 5: 30, 6: 150})
def reset_servos(): set_servo_angles({i: 90 for i in range(16)})
def sleep_fn(): set_servo_angles({0: 90, 1: 90, 2: 90, 3: 0, 4: 40, 5: 0, 6: 180})
def down_fn(): set_servo_angles({0: 90, 1: 90, 2: 90, 3: 180, 4: 40, 5: 0, 6: 180})
def standing_tall(): set_servo_angles({0: 90, 1: 90, 2: 90, 3: 180, 4: 170, 5: 0, 6: 180})

# Placeholder Arm/Head functions
def pan(): print("Command: PAN")
def binocular_eyes(): print("Command: BINOCULAR_EYES")
def squinting(): print("Command: SQUINTING")
def tank_treads(): print("Command: TANK_TREADS")
def crab_turn(): print("Command: CRAB_TURN")
def high_speed_roll(): print("Command: HIGH_SPEED_ROLL")
def lifting(): print("Command: LIFTING")
def crushing(): print("Command: CRUSHING")
def pointing(): print("Command: POINTING")
def hand_holding(): print("Command: HAND_HOLDING")
def tapping(): print("Command: TAPPING")

# --- Motor Driving Functions ---
def movef():
    if h:
        print("Driving: FORWARD")
        lgpio.gpio_write(h, IN1, 1)
        lgpio.gpio_write(h, IN2, 0)
        lgpio.gpio_write(h, IN3, 1)
        lgpio.gpio_write(h, IN4, 0)

def moveb():
    if h:
        print("Driving: BACKWARD")
        lgpio.gpio_write(h, IN1, 0)
        lgpio.gpio_write(h, IN2, 1)
        lgpio.gpio_write(h, IN3, 0)
        lgpio.gpio_write(h, IN4, 1)

def movel():
    if h:
        print("Driving: LEFT")
        lgpio.gpio_write(h, IN1, 0)
        lgpio.gpio_write(h, IN2, 1)
        lgpio.gpio_write(h, IN3, 1)
        lgpio.gpio_write(h, IN4, 0)

def mover():
    if h:
        print("Driving: RIGHT")
        lgpio.gpio_write(h, IN1, 1)
        lgpio.gpio_write(h, IN2, 0)
        lgpio.gpio_write(h, IN3, 0)
        lgpio.gpio_write(h, IN4, 1)

def movebl():
    if h:
        print("Driving: BACK-LEFT")
        lgpio.gpio_write(h, IN1, 0)
        lgpio.gpio_write(h, IN2, 0)
        lgpio.gpio_write(h, IN3, 0)
        lgpio.gpio_write(h, IN4, 1)

def movebr():
    if h:
        print("Driving: BACK-RIGHT")
        lgpio.gpio_write(h, IN1, 0)
        lgpio.gpio_write(h, IN2, 1)
        lgpio.gpio_write(h, IN3, 0)
        lgpio.gpio_write(h, IN4, 0)

def stopm():
    if h:
        print("Driving: STOP")
        lgpio.gpio_write(h, IN1, 0)
        lgpio.gpio_write(h, IN2, 0)
        lgpio.gpio_write(h, IN3, 0)
        lgpio.gpio_write(h, IN4, 0)

# Command map
command_map = {
    "HIDING": hiding, "NOD_NO": nod_no, "NOD_YES": nod_yes, "DANCE": dance,
    "RESET_BACK": reset_back_servos, "RESET": reset_servos, "SLEEP": sleep_fn,
    "DOWN": down_fn, "STANDING_TALL": standing_tall, "PAN": pan,
    "BINOCULAR_EYES": binocular_eyes, "SQUINTING": squinting,
    "TANK_TREADS": tank_treads, "CRAB_TURN": crab_turn,
    "HIGH_SPEED_ROLL": high_speed_roll, "LIFTING": lifting,
    "CRUSHING": crushing, "POINTING": pointing, "HAND_HOLDING": hand_holding,
    "TAPPING": tapping, "MOVEF": movef, "MOVEB": moveb, "MOVEL": movel,
    "MOVER": mover, "MOVEBL": movebl, "MOVEBR": movebr, "STOPM": stopm,
}

# --- Flask Web App ---

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/command")
def handle_command():
    cmd = request.args.get("cmd", "").upper()
    fn = command_map.get(cmd)
    if fn:
        fn()
        return f"Executed: {cmd}", 200
    return "Unknown command", 400

@app.route("/setspeed", methods=["POST"])
def set_speed():
    try:
        duty = int(request.form.get("speed", 100))
        duty = max(0, min(100, duty))
        set_motor_speed(duty)
        return f"Speed updated to {duty}%", 200
    except Exception as e:
        return f"Error: {e}", 400

# --- HTML TEMPLATE with Slider ---
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wall-E Control Panel</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom styles for a better look and feel */
        body {
            font-family: 'Inter', sans-serif;
            -webkit-tap-highlight-color: transparent; /* Disable tap highlight on mobile */
        }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

        .control-panel {
            display: grid;
            grid-template-areas:
                "title title"
                "actions dpad"
                "misc dpad";
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr auto;
            gap: 2rem;
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: #1f2937; /* gray-800 */
            border-radius: 1.5rem;
            box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        }

        @media (max-width: 768px) {
            .control-panel {
                grid-template-areas:
                    "title"
                    "dpad"
                    "actions"
                    "misc";
                grid-template-columns: 1fr;
                padding: 1.5rem;
                margin: 1rem;
            }
        }

        .panel-title { grid-area: title; }
        .actions-grid { grid-area: actions; }
        .misc-grid { grid-area: misc; }
        .dpad-container { grid-area: dpad; }

        /* D-Pad specific grid layout */
        .dpad {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 0.5rem;
            width: 100%;
            max-width: 250px;
            margin: auto;
            aspect-ratio: 1 / 1;
        }
        
        /* Arrow styles using SVG for sharpness */
        .arrow { width: 60%; height: 60%; }
        .arrow-up { transform: rotate(0deg); }
        .arrow-down { transform: rotate(180deg); }
        .arrow-left { transform: rotate(-90deg); }
        .arrow-right { transform: rotate(90deg); }
        .arrow-bl { transform: rotate(-135deg); }
        .arrow-br { transform: rotate(135deg); }

        /* Button base styles */
        .btn {
            display: flex;
            justify-content: center;
            align-items: center;
            border: none;
            border-radius: 0.75rem;
            color: white;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        }
        .btn:active {
            transform: scale(0.95);
            box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.2);
        }
        
        /* Action button styles */
        .btn-action {
            background-color: #4f46e5; /* Indigo-600 */
            padding: 0.75rem;
            font-size: 0.875rem;
        }
        .btn-action:hover { background-color: #4338ca; /* Indigo-700 */ }
        
        /* D-pad button styles */
        .btn-dpad {
            background-color: #374151; /* Gray-700 */
            border: 2px solid #4b5563; /* Gray-600 */
        }
        .btn-dpad:hover { background-color: #4b5563; /* Gray-600 */ }
        
        .btn-stop {
            background-color: #dc2626; /* Red-600 */
            border-color: #f87171; /* Red-400 */
        }
        .btn-stop:hover { background-color: #b91c1c; /* Red-700 */ }

        /* Positioning D-pad buttons */
        .dpad-up { grid-column: 2; grid-row: 1; }
        .dpad-down { grid-column: 2; grid-row: 3; }
        .dpad-left { grid-column: 1; grid-row: 2; }
        .dpad-right { grid-column: 3; grid-row: 2; }
        .dpad-bl { grid-column: 1; grid-row: 3; }
        .dpad-br { grid-column: 3; grid-row: 3; }
        .dpad-stop { grid-column: 2; grid-row: 2; }
        .dpad-empty { grid-column: 1; grid-row: 1; }
        .dpad-empty2 { grid-column: 3; grid-row: 1; }
    </style>
</head>
<body class="bg-gray-900 text-gray-200">

    <div class="control-panel">
        <div class="panel-title text-center">
            <h1 class="text-4xl font-bold text-white tracking-wider">WALL-E</h1>
            <p class="text-indigo-400">Remote Control Panel</p>
        </div>

        <div class="actions-grid flex flex-col gap-4">
            <h2 class="text-xl font-bold text-gray-300 border-b-2 border-gray-700 pb-2">Main Actions</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                <button class="btn btn-action" onclick="sendCmd('DANCE')">Dance</button>
                <button class="btn btn-action" onclick="sendCmd('STANDING_TALL')">Stand Tall</button>
                <button class="btn btn-action" onclick="sendCmd('SLEEP')">Sleep</button>
                <button class="btn btn-action" onclick="sendCmd('DOWN')">Down</button>
                <button class="btn btn-action" onclick="sendCmd('NOD_YES')">Nod Yes</button>
                <button class="btn btn-action" onclick="sendCmd('NOD_NO')">Nod No</button>
                <button class="btn btn-action col-span-2 md:col-span-3 bg-red-600 hover:bg-red-700" onclick="sendCmd('RESET')">Reset All</button>
            </div>
        </div>
        
        <div class="misc-grid flex flex-col gap-4">
            <h2 class="text-xl font-bold text-gray-300 border-b-2 border-gray-700 pb-2">Arm & Head Controls</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                <button class="btn btn-action" onclick="sendCmd('LIFTING')">Lift</button>
                <button class="btn btn-action" onclick="sendCmd('CRUSHING')">Crush</button>
                <button class="btn btn-action" onclick="sendCmd('POINTING')">Point</button>
                <button class="btn btn-action" onclick="sendCmd('TAPPING')">Tap</button>
                <button class="btn btn-action" onclick="sendCmd('PAN')">Pan</button>
                <button class="btn btn-action" onclick="sendCmd('SQUINTING')">Squint</button>
            </div>
        </div>

        <div class="dpad-container flex flex-col gap-4">
            <h2 class="text-xl font-bold text-gray-300 border-b-2 border-gray-700 pb-2 text-center">Movement</h2>
            <div class="dpad">
                <div class="dpad-empty"></div>
                <button class="btn btn-dpad dpad-up" onmousedown="startCmd('MOVEF')" onmouseup="stopCmd()" ontouchstart="startCmd('MOVEF')" ontouchend="stopCmd()">
                    <svg class="arrow arrow-up" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
                </button>
                <div class="dpad-empty2"></div>
                <button class="btn btn-dpad dpad-left" onmousedown="startCmd('MOVEL')" onmouseup="stopCmd()" ontouchstart="startCmd('MOVEL')" ontouchend="stopCmd()">
                    <svg class="arrow arrow-left" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
                </button>
                <button class="btn btn-dpad btn-stop dpad-stop" onclick="sendCmd('STOPM')">STOP</button>
                <button class="btn btn-dpad dpad-right" onmousedown="startCmd('MOVER')" onmouseup="stopCmd()" ontouchstart="startCmd('MOVER')" ontouchend="stopCmd()">
                    <svg class="arrow arrow-right" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
                </button>
                <button class="btn btn-dpad dpad-bl" onmousedown="startCmd('MOVEBL')" onmouseup="stopCmd()" ontouchstart="startCmd('MOVEBL')" ontouchend="stopCmd()">
                    <svg class="arrow arrow-bl" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
                </button>
                <button class="btn btn-dpad dpad-down" onmousedown="startCmd('MOVEB')" onmouseup="stopCmd()" ontouchstart="startCmd('MOVEB')" ontouchend="stopCmd()">
                    <svg class="arrow arrow-down" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
                </button>
                <button class="btn btn-dpad dpad-br" onmousedown="startCmd('MOVEBR')" onmouseup="stopCmd()" ontouchstart="startCmd('MOVEBR')" ontouchend="stopCmd()">
                    <svg class="arrow arrow-br" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
                </button>
            </div>
        </div>
    </div>

<script>
    let cmdInterval = null;

    // Function to send a single command to the server
    function sendCmd(cmd) {
        fetch('/command?cmd=' + cmd)
            .then(response => response.text())
            .then(text => console.log(text))
            .catch(err => console.error('Command failed:', err));
    }

    // Function to start sending a command repeatedly (for movement)
    function startCmd(cmd) {
        if (cmdInterval) {
            clearInterval(cmdInterval);
        }
        sendCmd(cmd);
        cmdInterval = setInterval(() => sendCmd(cmd), 200);
    }

    // Function to stop the repeating command and send a 'STOPM' command
    function stopCmd() {
        if (cmdInterval) {
            clearInterval(cmdInterval);
            cmdInterval = null;
        }
        sendCmd("STOPM");
    }
</script>

</body>
</html>
"""


# --- Run App ---
if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        print("\nShutting down, stopping motors and cleaning up GPIO.")
        if h:
            stopm()
            lgpio.gpio_write(h, ENA, 0)
            lgpio.gpio_write(h, ENB, 0)
            lgpio.gpiochip_close(h)