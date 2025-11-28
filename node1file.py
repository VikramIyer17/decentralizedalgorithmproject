from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Change this on each EC2 node
NODE_ID = "node1"   # node2 for the other instance

# ---- HEALTH CHECK ----
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"node": NODE_ID, "status": "alive"})


# ---- RECEIVE MESSAGE ----
@app.route("/receive", methods=["POST"])
def receive():
    data = request.json
    print(f"📩 {NODE_ID} received:", data)
    return jsonify({"received_by": NODE_ID, "ok": True})


# ---- SEND MESSAGE ----
@app.route("/send_message", methods=["POST"])
def send_message():
    content = request.json
    target_ip = content["target_ip"]
    message = content["message"]

    url = f"http://{target_ip}:5000/receive"

    response = requests.post(url, json={"from": NODE_ID, "msg": message})

    return jsonify({
        "sent_from": NODE_ID,
        "sent_to": target_ip,
        "response": response.json()
    })


if __name__ == "__main__":
    # IMPORTANT: Allow world access
    app.run(host="0.0.0.0", port=5000)
