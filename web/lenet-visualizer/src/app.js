import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Scene, Camera, and Renderer Initialization
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 20, 100);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(0, 20, 10);
scene.add(directionalLight);

// OrbitControls for camera movement
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minDistance = 10;
controls.maxDistance = 500;
controls.update();

// Image Upload Handling
document.getElementById('uploadButton').addEventListener('click', () => {
  const imageInput = document.getElementById('imageInput');
  if (!imageInput.files.length) {
    alert('Please select an image file.');
    return;
  }

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      console.log('Received activations:', data);
      renderActivations(data);
    })
    .catch(error => {
      console.error('Error uploading image:', error);
    });
});

// Render Activations
function renderActivations(data) {
  // Clear the scene before rendering new activations
  while (scene.children.length > 0) {
    scene.remove(scene.children[0]);
  }

  scene.add(ambientLight);
  scene.add(directionalLight);

  const layerSpacing = 10;
  let zOffset = 0;

  data.forEach((layer, layerIndex) => {
    if (Array.isArray(layer[0]) && Array.isArray(layer[0][0])) {
      const numChannels = layer[0][0].length;
      const gridSize = Math.ceil(Math.sqrt(numChannels));
      const featureMapWidth = layer[0].length;
      const totalLayerWidth = gridSize * (featureMapWidth + 2);

      for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
        render2DFeatureMap(layer, channelIndex, zOffset, totalLayerWidth);
      }
    } else if (Array.isArray(layer) && typeof layer[0] === 'number') {
      render1DActivations(layer, zOffset);
    }

    zOffset += layerSpacing;
  });

  animate();
}

// Render 2D Feature Maps for Convolutional Layers
function render2DFeatureMap(layer, channelIndex, zOffset, totalLayerWidth) {
  const height = layer.length;
  const width = layer[0].length;
  const planeSpacing = 2;

  const numChannels = layer[0][0].length;
  const gridSize = Math.ceil(Math.sqrt(numChannels));

  // Calculate row and column for the current feature map
  const row = Math.floor(channelIndex / gridSize);
  const col = channelIndex % gridSize;

  // Extract the feature map for the current channel
  const featureMap = layer.map(row => row.map(pixel => pixel[channelIndex]));
  const texture = createActivationTexture(featureMap, width, height);

  // Create a plane geometry and material to visualize the feature map
  const planeGeometry = new THREE.PlaneGeometry(width, height);
  const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);

  // Calculate position offsets
  const xOffset = -totalLayerWidth / 2 + col * (width + planeSpacing) + width / 2;
  const yOffset = -row * (height + planeSpacing);

  // Set the position of the plane and add it to the scene
  plane.position.set(xOffset, yOffset, -zOffset);
  scene.add(plane);
}

// Create Texture for 2D Activation Maps
function createActivationTexture(featureMap, width, height) {
  const scale = 255;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = featureMap[y][x] * scale;
      value = Math.min(Math.max(value, 0), 255);
      const index = (y * width + x) * 4;

      imageData.data[index] = value;       // Red
      imageData.data[index + 1] = value;   // Green
      imageData.data[index + 2] = value;   // Blue
      imageData.data[index + 3] = 255;     // Alpha
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return new THREE.CanvasTexture(canvas);
}


// Render 1D Activations for Fully Connected Layers
function render1DActivations(activations, zOffset) {
  const width = activations.length;
  const barHeight = 1;

  const texture = create1DActivationTexture(activations, width, barHeight);

  const planeGeometry = new THREE.PlaneGeometry(width, barHeight);
  const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);

  plane.position.set(0, 0, -zOffset);
  scene.add(plane);
}

// Create Texture for 1D Activation Maps
function create1DActivationTexture(activations, width, height) {
  const scale = 255;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  for (let x = 0; x < width; x++) {
    const value = Math.max(0, Math.min(1, activations[x]));
    const intensity = value * scale;

    for (let y = 0; y < height; y++) {
      const index = (y * width + x) * 4;
      imageData.data[index] = intensity;       // Red
      imageData.data[index + 1] = intensity;   // Green
      imageData.data[index + 2] = intensity;   // Blue
      imageData.data[index + 3] = 255;         // Alpha
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return new THREE.CanvasTexture(canvas);
}

// Animation Loop to Continuously Render the Scene
function animate() {
  requestAnimationFrame(animate);
  controls.update();  // Update OrbitControls for smooth interactions
  renderer.render(scene, camera);
}

// Start the Animation Loop
animate();
