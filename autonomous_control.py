import serial
import time
import signal
import sys
import csv
import os
import json

# Robot specific module imports
import tts_module
import vision_module
import thinking_module
import robot_actions

# --- Configuration ---
SERIAL_PORT = '/dev/tty.usbserial-0001'  # CHANGE THIS to your ESP32's serial port
BAUD_RATE = 115200
DATA_TIMEOUT = 1.0  # Seconds to wait for serial data
EXPECTED_COLUMNS = 16

# IPC Flag Files (must match wakeword_server.py)
WAKE_WORD_FLAG_FILE = "WAKE_WORD_DETECTED.flag"
LISTENING_COMPLETE_FLAG_FILE = "LISTENING_COMPLETE.flag"
USER_SPEECH_FILE = "user_speech.txt"
REQUEST_AUDIO_CAPTURE_FLAG = "REQUEST_AUDIO_CAPTURE.flag" # New flag for robot-initiated listening

# Global serial object
ser = None

# Robot States
STATE_IDLE = "IDLE" # Doing nothing, waiting for wakeword or initial start
STATE_AUTONOMOUS_NAV = "AUTONOMOUS_NAV" # Moving based on ESP32 PID
STATE_CRITICAL_OBSTACLE_HANDLER = "CRITICAL_OBSTACLE_HANDLER" # ESP32 stopped due to obstacle, Python taking over
STATE_SURVEY_MODE = "SURVEY_MODE" # Taking pictures (front, left, right)
STATE_AWAITING_LLM_DECISION = "AWAITING_LLM_DECISION" # Sent info to LLM, waiting for JSON response
STATE_PROCESSING_USER_COMMAND = "PROCESSING_USER_COMMAND" # Wakeword heard, user speech captured, sending to LLM
STATE_EXECUTING_LLM_DECISION = "EXECUTING_LLM_DECISION" # Performing actions from LLM JSON
STATE_AWAITING_REQUESTED_SPEECH = "AWAITING_REQUESTED_SPEECH" # Robot asked a question and is waiting for user speech

current_robot_state = STATE_IDLE
previous_robot_data = None # To store the last complete robot data packet
last_llm_decision = None # To store the last decision from the LLM

def cleanup_and_exit(sig=None, frame=None):
    """Gracefully close the serial port and exit."""
    global ser, current_robot_state
    print("\nCleaning up and exiting...")
    current_robot_state = STATE_IDLE # Stop any ongoing processes
    if ser and ser.is_open:
        try:
            robot_actions.stop_robot(ser) # Send a final stop command
            ser.close()
            print("Serial port closed.")
        except Exception as e:
            print(f"Error during serial cleanup: {e}")
    # Clear IPC flags (optional, server should also do this on its exit)
    if os.path.exists(WAKE_WORD_FLAG_FILE):
        try: os.remove(WAKE_WORD_FLAG_FILE)
        except OSError as e: print(f"Error removing flag file {WAKE_WORD_FLAG_FILE}: {e}")
    if os.path.exists(LISTENING_COMPLETE_FLAG_FILE):
        try: os.remove(LISTENING_COMPLETE_FLAG_FILE)
        except OSError as e: print(f"Error removing flag file {LISTENING_COMPLETE_FLAG_FILE}: {e}")
    if os.path.exists(USER_SPEECH_FILE):
        try: os.remove(USER_SPEECH_FILE)
        except OSError as e: print(f"Error removing speech file {USER_SPEECH_FILE}: {e}")
    if os.path.exists(REQUEST_AUDIO_CAPTURE_FLAG):
        try: os.remove(REQUEST_AUDIO_CAPTURE_FLAG)
        except OSError as e: print(f"Error removing flag file {REQUEST_AUDIO_CAPTURE_FLAG}: {e}")
    sys.exit(0)

def clear_flag_file(flag_path):
    """Safely remove a flag file."""
    if os.path.exists(flag_path):
        try:
            os.remove(flag_path)
            print(f"IPC: Cleared flag file {flag_path}")
        except OSError as e:
            print(f"Error removing flag file {flag_path}: {e}")

def connect_serial():
    """Establish serial connection with the ESP32."""
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=DATA_TIMEOUT)
        print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        time.sleep(2)  # Wait for ESP32 to reset
        ser.reset_input_buffer()
        return True
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        return False

def parse_data_line(line):
    """Parse a CSV line of sensor data from ESP32."""
    try:
        if isinstance(line, bytes):
            line = line.decode('utf-8').strip()
        else:
            line = line.strip()
        if not line: return None

        values = list(csv.reader([line]))[0]
        if len(values) != EXPECTED_COLUMNS:
            # print(f"Warning: Malformed data. Expected {EXPECTED_COLUMNS}, got {len(values)}. Data: '{line}'")
            return None
        
        data = {
            "L90": int(values[0]), "L45": int(values[1]), "F": int(values[2]), "R45": int(values[3]), "R90": int(values[4]),
            "sW_L90": float(values[5]), "sW_L45": float(values[6]), "sW_F": float(values[7]), "sW_R45": float(values[8]), "sW_R90": float(values[9]),
            "SteeringIn": float(values[10]), "PID_Out": float(values[11]),
            "L_Speed": int(values[12]), "R_Speed": int(values[13]),
            "BaseSpeed": int(values[14]), "CurSpeed": int(values[15]),
            "PID_Active_ESP": True # Placeholder, ESP32 doesn't send this yet, but useful for logic
        }
        # Actual PID active status can be inferred if CurSpeed > 0 and no manual command was just sent,
        # or if ESP32 sends it. For now, this is a simplification.
        # We can check if CurSpeed is 0 as an indicator of PID stopping or manual stop.
        return data
    except ValueError as e:
        # print(f"Error converting data: {e}. Line: '{line}'")
        return None
    except Exception as e:
        # print(f"An unexpected error during parsing: {e}. Line: '{line}'")
        return None


def initialize_systems():
    """Initialize TTS and Vision models."""
    print("Initializing Text-to-Speech system...")
    if not tts_module.initialize_tts():
        print("CRITICAL: Failed to initialize TTS. Speech functions will not work.")
        # Potentially exit or run in a degraded mode
    
    print("Initializing Vision system...")
    if not vision_module.initialize_vision_model():
        print("CRITICAL: Failed to initialize Vision model. Survey functions will not work.")
        # Potentially exit or run in a degraded mode
    
    print("All external systems initialized (or attempted).")

# --- Main Application Logic ---
def main_loop():
    global ser, current_robot_state, previous_robot_data, last_llm_decision
    
    print(f"\nStarting autonomous control loop. Initial state: {current_robot_state}")
    print("Press Ctrl+C to exit.")

    # Set initial ESP32 mode to autonomous if desired
    # robot_actions.set_autonomous_mode(ser, True) # Start in autonomous
    # current_robot_state = STATE_AUTONOMOUS_NAV # If starting autonomously

    last_data_print_time = time.time()
    last_wake_word_check_time = time.time()
    awaiting_speech_start_time = None # Timer for robot-requested speech

    while True:
        try:
            # --- Read Serial Data from ESP32 ---
            raw_line = None
            if ser and ser.in_waiting > 0:
                raw_line = ser.readline()
                # print(f"Raw from ESP32: {raw_line.strip()}") # Debug raw data
                robot_data = parse_data_line(raw_line)
                if robot_data:
                    previous_robot_data = robot_data # Always store the latest valid data
                    # Optional: Print data periodically
                    if time.time() - last_data_print_time > 2.0: # Print data every 2s
                        print(f"State: {current_robot_state} | ESP32 Data: F={robot_data['F']} L45={robot_data['L45']} R45={robot_data['R45']} | Speed L={robot_data['L_Speed']} R={robot_data['R_Speed']} Cur={robot_data['CurSpeed']}")
                        last_data_print_time = time.time()
                
                # Handle non-data messages (like PID Active status from ESP32 if implemented)
                elif raw_line:
                    try:
                        decoded_line = raw_line.decode('utf-8').strip()
                        if "BaseSpeed:" in decoded_line or "PID Active:" in decoded_line or "initialized." in decoded_line or "Failed to detect" in decoded_line or "TIMEOUT" in decoded_line:
                            print(f"ESP32 MSG: {decoded_line}")
                    except UnicodeDecodeError:
                        pass # Already handled by parse_data_line somewhat
            
            # --- IPC: Check for Wake Word Server Signals ---
            if time.time() - last_wake_word_check_time > 0.25: # Check every 250ms
                if os.path.exists(LISTENING_COMPLETE_FLAG_FILE):
                    print("IPC: LISTENING_COMPLETE_FLAG_FILE detected.")
                    if current_robot_state not in [STATE_AWAITING_LLM_DECISION, STATE_EXECUTING_LLM_DECISION]: # Avoid interrupting critical LLM tasks
                        robot_actions.stop_robot(ser) # Stop robot movement
                        current_robot_state = STATE_PROCESSING_USER_COMMAND
                    else:
                        print(f"IPC: In state {current_robot_state}, deferring user command processing.")
                    # Flag will be cleared after processing the command

                elif os.path.exists(WAKE_WORD_FLAG_FILE):
                    print("IPC: WAKE_WORD_FLAG_FILE detected.")
                    if current_robot_state not in [STATE_AWAITING_LLM_DECISION, STATE_EXECUTING_LLM_DECISION, STATE_PROCESSING_USER_COMMAND, STATE_SURVEY_MODE, STATE_CRITICAL_OBSTACLE_HANDLER]:
                        print("IPC: Wake word acknowledged. Robot stopping. Server is now listening for command.")
                        robot_actions.stop_robot(ser) # Stop the robot
                        current_robot_state = STATE_IDLE # Or a new state like AWAITING_USER_SPEECH
                    else:
                        print(f"IPC: In state {current_robot_state}, wake word ignored by main brain for now.")
                    # Wakeword server clears its own WAKE_WORD_FLAG_FILE after starting to listen for command.
                    # Or we can clear it here if the server logic changes.
                    # For now, assume server handles it.
                    clear_flag_file(WAKE_WORD_FLAG_FILE) # Main brain now clears the flag
                last_wake_word_check_time = time.time()

            # --- Robot State Machine ---
            if current_robot_state == STATE_IDLE:
                # Waiting for a wake word or an event to trigger another state.
                # Robot will now wait for a voice command (via IPC) to transition.
                # The automatic transition to AUTONOMOUS_NAV has been removed.
                pass # Stay idle until an IPC event or other state change occurs

            elif current_robot_state == STATE_AUTONOMOUS_NAV:
                if previous_robot_data:
                    # Check for critical stop condition triggered by ESP32's TOF logic
                    # ESP32's CurrentSpeed becomes 0 when it stops due to obstacle (and PID was active)
                    # We need to ensure this wasn't a manual 'x' stop or PID toggle.
                    # This logic assumes ESP32 sets CurSpeed to 0 when its internal avoidance stops it.
                    if previous_robot_data["CurSpeed"] == 0 and previous_robot_data.get("PID_Active_ESP", True):
                        # How to differentiate from a manual 'x' command or 'p' toggle from Python side?
                        # For now, assume if Python didn't just send 'x' or 'p', and CurSpeed is 0, ESP32 stopped itself.
                        # This needs robust check. Let's assume for now this is the ESP32's autonomous stop.
                        print("State AUTONOMOUS_NAV: ESP32 reported CurSpeed = 0. Obstacle detected by ESP32.")
                        current_robot_state = STATE_CRITICAL_OBSTACLE_HANDLER
                # If wake word detected, state will change via IPC check above.

            elif current_robot_state == STATE_CRITICAL_OBSTACLE_HANDLER:
                print("State CRITICAL_OBSTACLE_HANDLER: Initiating backup and survey.")
                robot_actions.backup_robot(ser, duration_seconds=1.5)
                # backup_robot already sends a stop command after backup.
                current_robot_state = STATE_SURVEY_MODE
                # Clear any pending LLM decision from a previous cycle if any
                last_llm_decision = None 

            elif current_robot_state == STATE_SURVEY_MODE:
                print("State SURVEY_MODE: Starting visual survey.")
                descriptions = {"front": "Error during front view", "left": "Error during left view", "right": "Error during right view"}

                # 1. Front View
                tts_module.speak("Let me take a look around. First, what's in front?")
                img_front = vision_module.capture_image()
                if img_front:
                    res_front = vision_module.analyze_image(img_front, prompt="Describe the scene in 3 sentences.")
                    descriptions["front"] = res_front.get('description', descriptions["front"])
                    print(f"Survey - Front: {descriptions['front']}")
                else: print("Survey - Front: Failed to capture image.")
                time.sleep(0.5)

                # 2. Left View
                tts_module.speak("Now, to my left.")
                robot_actions.turn_robot(ser, 'left', duration_seconds=2.0)
                img_left = vision_module.capture_image()
                if img_left:
                    res_left = vision_module.analyze_image(img_left, prompt="Describe the scene in 3 sentences.")
                    descriptions["left"] = res_left.get('description', descriptions["left"])
                    print(f"Survey - Left: {descriptions['left']}")
                else: print("Survey - Left: Failed to capture image.")
                time.sleep(0.5)

                # 3. Right View 
                tts_module.speak("And finally, to my right.")
                robot_actions.turn_robot(ser, 'right', duration_seconds=4) # Turn right
                img_right = vision_module.capture_image()
                if img_right:
                    res_right = vision_module.analyze_image(img_right, prompt="Describe the scene in 3 sentences.")
                    descriptions["right"] = res_right.get('description', descriptions["right"])
                    print(f"Survey - Right: {descriptions['right']}")
                else: print("Survey - Right: Failed to capture image.")
                time.sleep(0.5)
                
                # Return to roughly center (optional, or let LLM decide next turn)
                # tts_module.speak("Okay, I've had a good look around.")
                robot_actions.turn_robot(ser, 'left', duration_seconds=2) # Attempt to re-center
                robot_actions.stop_robot(ser)

                print("State SURVEY_MODE: Survey complete. Requesting LLM decision.")
                last_llm_decision = thinking_module.get_decision_for_survey(
                    descriptions["front"],
                    descriptions["left"],
                    descriptions["right"]
                )
                current_robot_state = STATE_EXECUTING_LLM_DECISION

            elif current_robot_state == STATE_PROCESSING_USER_COMMAND:
                print("State PROCESSING_USER_COMMAND: Reading user speech.")
                user_speech_text = ""
                try:
                    with open(USER_SPEECH_FILE, 'r') as f:
                        user_speech_text = f.read().strip()
                    os.remove(USER_SPEECH_FILE) # Clean up file
                except Exception as e:
                    print(f"Error reading or deleting user speech file: {e}")
                    tts_module.speak("I had trouble understanding what you said.")
                    clear_flag_file(LISTENING_COMPLETE_FLAG_FILE) # Ensure flag is cleared before changing state
                    current_robot_state = STATE_IDLE
                    continue

                if user_speech_text:
                    formatted_command = f"a voice addressing you has said \"{user_speech_text}\" how do you respond?"
                    print(f"Sending to LLM for user command: {formatted_command}")
                    last_llm_decision = thinking_module.get_decision_for_user_command(formatted_command)
                    current_robot_state = STATE_EXECUTING_LLM_DECISION
                else:
                    print("User speech was empty.")
                    tts_module.speak("I didn't catch that, please try again after the wake word.")
                    current_robot_state = STATE_IDLE
                
                clear_flag_file(LISTENING_COMPLETE_FLAG_FILE) # Always clear the flag after attempting to process
            
            elif current_robot_state == STATE_EXECUTING_LLM_DECISION:
                print(f"State EXECUTING_LLM_DECISION: Processing decision: {json.dumps(last_llm_decision, indent=2)}")
                if not last_llm_decision or 'error' in last_llm_decision:
                    tts_module.speak(last_llm_decision.get("think", "I had a problem with my thinking process. I'll just stop for now."))
                    if last_llm_decision.get("speak"): # If error response includes speak
                         tts_module.speak(last_llm_decision["speak"])
                    robot_actions.stop_robot(ser)
                    current_robot_state = STATE_IDLE # Default to idle on error
                    last_llm_decision = None
                    continue

                # Execute actions in order specified by LLM (if keys exist)
                if last_llm_decision.get("think"): # Always good to log the thought
                    print(f"LLM Think: {last_llm_decision['think']}")
                
                if last_llm_decision.get("speak"):
                    tts_module.speak(last_llm_decision["speak"])
                    time.sleep(0.2) # Small pause after speaking
                
                if last_llm_decision.get("stop") is True:
                    robot_actions.stop_robot(ser)
                    current_robot_state = STATE_IDLE
                    last_llm_decision = None
                    continue # Stop further actions in this decision

                if last_llm_decision.get("survey") is True:
                    current_robot_state = STATE_SURVEY_MODE
                    last_llm_decision = None # Clear decision as survey will generate a new one
                    continue

                if last_llm_decision.get("turn_left") is True:
                    tts_module.speak("Okay, turning left.")
                    robot_actions.turn_robot(ser, 'left', 1.0) # Default turn, LLM could specify duration in future
                    # After turn, usually a survey or re-evaluation is needed.
                    # For now, let's assume LLM will chain 'survey' if needed or go to autonomous.

                if last_llm_decision.get("turn_right") is True:
                    tts_module.speak("Alright, turning right.")
                    robot_actions.turn_robot(ser, 'right', 1.0)
                
                # change_directive is for future complex tasks, for now, just acknowledge
                if last_llm_decision.get("change_directive"):
                    tts_module.speak(f"Okay, I will now try to: {last_llm_decision['change_directive']}")
                    # Implementation of actually changing directive would go here.
                    # For now, it might just influence future 'think' prompts implicitly via history.

                if last_llm_decision.get("move_forward_autonomously") is True:
                    tts_module.speak("Moving forward autonomously.")
                    robot_actions.set_autonomous_mode(ser, True)
                    current_robot_state = STATE_AUTONOMOUS_NAV
                elif last_llm_decision.get("listen_for_response") is True:
                    # If LLM provided speech, it would have been spoken already by the generic handler.
                    # Now, specifically handle the listening initiation.
                    tts_module.speak("I'm listening.")
                    try:
                        with open(REQUEST_AUDIO_CAPTURE_FLAG, 'w') as f:
                            f.write('1') # Create the flag file
                        print(f"IPC: Created flag file {REQUEST_AUDIO_CAPTURE_FLAG}")
                        current_robot_state = STATE_AWAITING_REQUESTED_SPEECH
                        awaiting_speech_start_time = time.time() # Start timer
                    except Exception as e:
                        print(f"Error creating {REQUEST_AUDIO_CAPTURE_FLAG}: {e}")
                        tts_module.speak("I have a problem setting up my listener.")
                        current_robot_state = STATE_IDLE
                else:
                    # If no explicit move command, and not stopped/surveying, what to do?
                    # Default to IDLE or AUTONOMOUS_NAV if no other action taken.
                    # If an action like turn was taken, it might be better to go to survey or idle to re-evaluate.
                    if not (last_llm_decision.get("turn_left") or last_llm_decision.get("turn_right")):
                        # If no movement actions were taken from LLM and not stopping/surveying
                        # Revert to autonomous nav or idle.
                        print("LLM decision executed, no explicit next movement. Reverting to autonomous navigation.")
                        robot_actions.set_autonomous_mode(ser, True)
                        current_robot_state = STATE_IDLE # Change to IDLE to allow pause/thinking
                    else:
                        # After a turn, probably best to re-evaluate or let LLM explicitly say move_forward_autonomously
                        print("LLM turn executed. Consider surveying or explicit LLM instruction to move.")
                        current_robot_state = STATE_IDLE # Go to idle to re-evaluate or wait for next LLM command cycle

                last_llm_decision = None # Clear decision after execution
            
            elif current_robot_state == STATE_AWAITING_REQUESTED_SPEECH:
                if os.path.exists(LISTENING_COMPLETE_FLAG_FILE):
                    print(f"IPC: {LISTENING_COMPLETE_FLAG_FILE} detected during AWAITING_REQUESTED_SPEECH.")
                    current_robot_state = STATE_PROCESSING_USER_COMMAND
                    awaiting_speech_start_time = None # Reset timer
                elif awaiting_speech_start_time and (time.time() - awaiting_speech_start_time > 20.0): # 20 second timeout
                    print("Timeout waiting for requested speech.")
                    tts_module.speak("I didn't hear a response. Going back to what I was doing.")
                    clear_flag_file(REQUEST_AUDIO_CAPTURE_FLAG) # Clean up our request flag
                    current_robot_state = STATE_IDLE # Or a previous state if more context is kept
                    awaiting_speech_start_time = None # Reset timer

            # Small delay to prevent high CPU usage if no blocking calls are made
            time.sleep(0.05)

        except serial.SerialException as e:
            print(f"Serial error: {e}. Attempting to reconnect...")
            if ser and ser.is_open: ser.close()
            time.sleep(3)
            if not connect_serial():
                print("Reconnect failed. Exiting.")
                cleanup_and_exit()
            else:
                if ser: ser.reset_input_buffer()
        except KeyboardInterrupt:
            print("Main loop interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred in main_loop: {e}")
            # Consider which state to revert to on general error, maybe IDLE
            # current_robot_state = STATE_IDLE
            tts_module.speak("Oh dear, something went wrong with my main functions.")
            time.sleep(1) # Avoid rapid error logging

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # Clear any lingering IPC files from a previous run
    for f in [WAKE_WORD_FLAG_FILE, LISTENING_COMPLETE_FLAG_FILE, USER_SPEECH_FILE, REQUEST_AUDIO_CAPTURE_FLAG]:
        if os.path.exists(f):
            try: os.remove(f)
            except OSError as e: print(f"Could not remove old IPC file {f}: {e}")

    initialize_systems() # Initialize TTS, Vision

    if connect_serial():
        # Start in IDLE state, it will transition to AUTONOMOUS_NAV if conditions are met
        current_robot_state = STATE_IDLE 
        # Or uncomment below to start directly in autonomous mode:
        # robot_actions.set_autonomous_mode(ser, True)
        # current_robot_state = STATE_AUTONOMOUS_NAV
        main_loop()
    
    cleanup_and_exit()