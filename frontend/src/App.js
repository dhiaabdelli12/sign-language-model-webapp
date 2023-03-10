import './App.css';
import { useEffect, useRef } from 'react';


const ws = new WebSocket("ws://localhost:8000/videofeed");

ws.onopen = () => {
  console.log("WebSocket connection established");
};

ws.onmessage = (event) => {
  console.log("Received message:", event.data);
};

ws.onerror = (event) => {
  console.error("WebSocket error:", event);
};

ws.onclose = (event) => {
  console.log("WebSocket connection closed with code:", event.code);
};


function WebCamComponent() {

  const canvasRef = useRef(null);
  const videoRef = useRef(null);

  useEffect(() => {

    const setupCamera = async () => {
      const constraints = {
        video: true,
        audio: false
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoRef.current.srcObject = stream;
        videoRef.current.play();

      } catch (error) {
        console.log(error);
      }



    }

    const captureFrame = async (video, canvas) => {
      const context = canvas.getContext('2d');
      context.drawImage(video.current, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      const frame = dataUrl.substring(dataUrl.indexOf(",") + 1);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(frame);
      } else {
        console.error('WebSocket is not open.');
      }
    }

    const startCapturing = async () => {
      const canvas = canvasRef.current;
      setupCamera();

      setInterval(() => {
        captureFrame(videoRef, canvas);
      }, 2000);
    }

    startCapturing();



  }, [])


  return (
    <>
      <h1>hello</h1>
      <video ref={videoRef} width={1280} height={720} />
      <canvas ref={canvasRef} width={1280} height={720} />
    </>
  );
}

export default WebCamComponent;
