import socketio

sio = socketio.Client()

@sio.on('training_event')
def on_event(data):
    print(f"📡 {data['type']}: Reward={data['data'].get('reward', 'N/A')}")

sio.connect('http://localhost:8000')
sio.wait()
