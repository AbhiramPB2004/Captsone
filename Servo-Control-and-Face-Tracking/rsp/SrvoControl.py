from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/move_servo', methods=['POST'])
def move_servo():
    data = request.get_json()
    direction = data.get('direction')
    step = int(data.get('step', 1))

    # Implement actual servo logic here
    print(f"Moving servo {direction} by {step} steps")

    # Example response
    return jsonify({"status": "success", "direction": direction, "step": step})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
