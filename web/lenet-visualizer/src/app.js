// Import Three.js and OrbitControls
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';


// Initialize the scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 20, 100);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add ambient and directional lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(0, 20, 10);
scene.add(directionalLight);

// Initialize OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // Adds inertia for smoother controls
controls.dampingFactor = 0.1;
controls.minDistance = 10;
controls.maxDistance = 500;
controls.target.set(0, 0, 0);
controls.update();

// Load the JSON data and render activations
fetch('../activations/lenet_layer_output.json')
  .then(response => response.json())
  .then(data => {
    renderActivations(data);
  })
  .catch(error => console.error('Error loading JSON:', error));

// Render activations for each layer
function renderActivations(data) {
  const layerSpacing = 10;
  let zOffset = 0;

  data.forEach((layer, layerIndex) => {
    if (Array.isArray(layer[0]) && Array.isArray(layer[0][0])) {
      // 2D activations (Height x Width x Channels)
      const numChannels = layer[0][0].length;
      const gridSize = Math.ceil(Math.sqrt(numChannels));
      const featureMapWidth = layer[0].length;
      const totalLayerWidth = gridSize * (featureMapWidth + 2); // Width including spacing

      for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
        render2DFeatureMap(layer, channelIndex, zOffset, totalLayerWidth);
      }
    } else if (Array.isArray(layer) && typeof layer[0] === 'number') {
      // 1D activations (e.g., fully connected layers)
      const totalLayerWidth = layer.length;
      render1DActivations(layer, zOffset, totalLayerWidth);
    } else {
      console.error(`Unknown layer format at index ${layerIndex}:`, layer);
    }

    zOffset += layerSpacing;
  });

  animate();
}


// Render 2D feature maps for convolutional layers
// Render 2D feature maps for convolutional layers in a grid layout
function render2DFeatureMap(layer, channelIndex, zOffset, totalLayerWidth) {
  const height = layer.length;
  const width = layer[0].length;

  const numChannels = layer[0][0].length;
  const gridSize = Math.ceil(Math.sqrt(numChannels));
  const planeSpacing = 2;

  // Calculate row and column indices for the current channel
  const row = Math.floor(channelIndex / gridSize);
  const col = channelIndex % gridSize;

  // Create the feature map for the current channel
  const featureMap = layer.map(row => row.map(pixel => pixel[channelIndex]));
  const texture = createActivationTexture(featureMap, width, height, 4);

  const planeGeometry = new THREE.PlaneGeometry(width, height);
  const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);

  // Calculate the x-offset to center the entire layer
  const xOffset = -totalLayerWidth / 2 + (col * (width + planeSpacing)) + width / 2;

  // Position the plane based on the grid layout and center alignment
  plane.position.set(xOffset, -row * (height + planeSpacing), -zOffset);
  scene.add(plane);
}


// Render 1D activations for fully connected or flattened layers
function render1DActivations(activations, zOffset, totalLayerWidth) {
  const barHeight = 1; // Fixed height of the bar
  const width = activations.length; // Width is equal to the number of activations

  // Create a texture representing the activations
  const texture = create1DActivationTexture(activations, width, barHeight);

  // Create a plane geometry to map the texture
  const planeGeometry = new THREE.PlaneGeometry(width, barHeight);
  const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeometry, planeMaterial);

  // Center the plane horizontally
  const xOffset = -totalLayerWidth / 2 + width / 2;

  // Position the plane
  plane.position.set(xOffset, 0, -zOffset);
  scene.add(plane);

  console.log(`Rendered 1D activation layer as a horizontal bar.`);
}


// Create texture for 2D activation map
function createActivationTexture(featureMap, width, height) {
  const scale = 255; // Scale factor to amplify small values
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = featureMap[y][x] * scale;
      value = Math.min(Math.max(value, 0), 255); // Clamp value between 0 and 255
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

function create1DActivationTexture(activations, width, height) {
  const scale = 255; // Scale factor to convert activation values to 0-255 range

  // Create a canvas for the texture
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  // Fill the canvas with activation values
  for (let x = 0; x < width; x++) {
    const value = Math.max(0, Math.min(1, activations[x])); // Clamp values between 0 and 1
    const intensity = value * scale; // Convert to 0-255 range

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

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update(); // Update the controls for smooth interactions
  renderer.render(scene, camera);
}
