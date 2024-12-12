import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

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

const pointLight = new THREE.PointLight(0xffffff, 0.8);
pointLight.position.set(20, 50, 20);
scene.add(pointLight);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
hemiLight.position.set(0, 50, 0);
scene.add(hemiLight);

// Enable shadow maps
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// Configure directional light for shadows
directionalLight.castShadow = true;
directionalLight.shadow.mapSize.width = 1024;
directionalLight.shadow.mapSize.height = 1024;
scene.add(directionalLight);

scene.background = new THREE.Color(0x00003b);


// OrbitControls for camera movement
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minDistance = 10;
controls.maxDistance = 500;
controls.update();

// Function to render the scene initially
function initialRender() {
  renderer.render(scene, camera);
}

// Call initialRender to display the background and lights
initialRender();

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
      renderInputImage(URL.createObjectURL(imageInput.files[0]), 0); // Display input image at zOffset 0
      renderActivations(data);
    })
    .catch(error => {
      console.error('Error uploading image:', error);
    });
});



function renderActivations(response) {
  const data = response.activations;
  const layerNames = response.layer_names;

  // Clear the scene before rendering new activations
  while (scene.children.length > 0) {
    scene.remove(scene.children[0]);
  }

  // Clear the layer tags list and add the reset button
  const layerTagsList = document.getElementById('layerTagsList');
  layerTagsList.innerHTML = '';

  // Add a Reset button
  const resetButton = document.createElement('button');
  resetButton.textContent = 'Show All Layers';
  resetButton.style.marginBottom = '10px';
  resetButton.style.padding = '5px 10px';
  resetButton.style.cursor = 'pointer';
  resetButton.addEventListener('click', () => {
    showAllLayers(layerMeshes, inputImageMesh);
  });
  layerTagsList.appendChild(resetButton);

  // Add input image at the beginning
  const initialZOffset = 0;
  const inputImageMesh = renderInputImage(URL.createObjectURL(imageInput.files[0]), initialZOffset, 50);

  // Initial spacing after the input image
  let zOffset = initialZOffset + 20;
  const layerMeshes = []; // Array to store layer meshes for toggling visibility

  data.forEach((layer, layerIndex) => {
    // Add layer tag to the sidebar
    const layerName = layerNames[layerIndex] || `Layer ${layerIndex + 1}`;
    const listItem = document.createElement('li');
    listItem.textContent = `${layerIndex + 1}: ${layerName}`;
    listItem.style.cursor = 'pointer';
    listItem.style.padding = '5px';
    listItem.style.borderBottom = '1px solid #444';

    // Add click event to toggle visibility of the layer
    listItem.addEventListener('click', () => {
      toggleLayerVisibility(layerIndex, layerMeshes, inputImageMesh, listItem);
    });

    layerTagsList.appendChild(listItem);

    // Create a label with the layer index above the layer
    createLayerIndexLabel(layerIndex + 1, zOffset);

    let currentMeshes = [];
    if (Array.isArray(layer[0]) && Array.isArray(layer[0][0])) {
      const numChannels = layer[0][0].length;
      const gridSize = Math.ceil(Math.sqrt(numChannels));
      const featureMapWidth = layer[0].length;
      const totalLayerWidth = gridSize * (featureMapWidth + 2);

      for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
        const mesh = render2DFeatureMap(layer, channelIndex, zOffset, totalLayerWidth);
        currentMeshes.push(mesh);
      }
    } else if (Array.isArray(layer) && typeof layer[0] === 'number') {
      const mesh = render1DActivations(layer, zOffset);
      currentMeshes.push(mesh);
    }

    layerMeshes.push(currentMeshes);

    zOffset += 20; // Uniform spacing between layers
  });

  animate();
}

function toggleLayerVisibility(selectedIndex, layerMeshes, inputImageMesh, selectedTag) {
  // Hide the input image
  if (inputImageMesh) inputImageMesh.visible = false;

  // Hide all layers except the selected one
  layerMeshes.forEach((meshes, index) => {
    meshes.forEach(mesh => {
      mesh.visible = index === selectedIndex;
    });
  });

  // Hide all layer tags and show only the selected one
  const layerTagsList = document.getElementById('layerTagsList');
  Array.from(layerTagsList.children).forEach(tag => {
    if (tag !== selectedTag && tag.tagName === 'LI') {
      tag.style.display = 'none';
    }
  });
}


// Render 2D Feature Maps for Convolutional Layers
function render2DFeatureMap(layer, channelIndex, zOffset, totalLayerWidth) {
  const height = layer.length;
  const width = layer[0].length;
  const voxelSize = 1;
  const planeSpacing = 2;

  const numChannels = layer[0][0].length;
  const gridSize = Math.ceil(Math.sqrt(numChannels));

  const row = Math.floor(channelIndex / gridSize);
  const col = channelIndex % gridSize;

  const xOffset = -totalLayerWidth / 2 + col * (width + planeSpacing) + width / 2;
  const yOffset = -row * (height + planeSpacing);

  const voxelGeometry = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);
  const voxelMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9 });
  const voxelCount = width * height;
  const instancedMesh = new THREE.InstancedMesh(voxelGeometry, voxelMaterial, voxelCount);

  let index = 0;
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const activationValue = layer[y][x][channelIndex];
      const voxelColor = getActivationColor(activationValue);

      // Set position
      dummy.position.set(xOffset + x * voxelSize, yOffset - y * voxelSize, -zOffset);
      dummy.updateMatrix();

      // Set color and apply to instance
      instancedMesh.setMatrixAt(index, dummy.matrix);
      instancedMesh.setColorAt(index, voxelColor);
      index++;
    }
  }

  instancedMesh.instanceMatrix.needsUpdate = true;
  instancedMesh.instanceColor.needsUpdate = true;

  scene.add(instancedMesh);
  return instancedMesh;
}



function getActivationColor(value) {
  // Normalize the value between 0 and 1
  const normalizedValue = Math.min(Math.max(value, 0), 1);

  // Use a heatmap color scheme (red -> yellow -> green -> blue)
  const colorScale = new THREE.Color();
  colorScale.setHSL(0.7 * (1 - normalizedValue), 1.0, 0.5); // Hue based on value

  return colorScale;
}




// Render 1D Activations for Fully Connected Layers
function render1DActivations(activations, zOffset) {
  const voxelSize = 0.5;  // Smaller voxel size
  const spacing = 0.6;    // More spacing between voxels
  const voxelGeometry = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);
  const voxelMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9 });
  const voxelCount = activations.length;
  const instancedMesh = new THREE.InstancedMesh(voxelGeometry, voxelMaterial, voxelCount);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  activations.forEach((value, index) => {
    const voxelColor = getActivationColor(value);

    dummy.position.set(index * spacing, 0, -zOffset);
    dummy.updateMatrix();

    instancedMesh.setMatrixAt(index, dummy.matrix);
    instancedMesh.setColorAt(index, voxelColor);
  });

  instancedMesh.instanceMatrix.needsUpdate = true;
  instancedMesh.instanceColor.needsUpdate = true;

  // Center the voxel group
  instancedMesh.position.x = -(voxelCount * spacing) / 2 + voxelSize / 2;

  scene.add(instancedMesh);
  return instancedMesh;
}


function renderInputImage(imageUrl, zOffset, maxSize = 50) {
  const textureLoader = new THREE.TextureLoader();
  const plane = new THREE.Mesh(); // Placeholder mesh to be returned

  textureLoader.load(imageUrl, (texture) => {
    const originalWidth = texture.image.width;
    const originalHeight = texture.image.height;

    // Calculate scaling factor to fit within maxSize while preserving the aspect ratio
    const scaleFactor = maxSize / Math.max(originalWidth, originalHeight);
    const targetWidth = originalWidth * scaleFactor;
    const targetHeight = originalHeight * scaleFactor;

    // Create a plane with the scaled width and height
    const planeGeometry = new THREE.PlaneGeometry(targetWidth, targetHeight);
    const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
    plane.geometry = planeGeometry;
    plane.material = planeMaterial;

    // Position the plane
    plane.position.set(0, -targetHeight / 2, -zOffset);
    scene.add(plane);
  });

  return plane; // Return the created mesh
}

function createLayerIndexLabel(index, zOffset) {
  const loader = new FontLoader();
  loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', (font) => {
    const geometry = new TextGeometry(`${index}`, {
      font: font,
      size: 3,
      height: 0.2,
    });

    const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const textMesh = new THREE.Mesh(geometry, material);

    // Position the index label above the layer
    textMesh.position.set(0, 25, -zOffset); // Adjust y-position to place above the layer
    scene.add(textMesh);
  });
}

function showAllLayers(layerMeshes, inputImageMesh) {
  // Show the input image
  if (inputImageMesh) inputImageMesh.visible = true;

  // Show all layers
  layerMeshes.forEach(meshes => {
    meshes.forEach(mesh => {
      mesh.visible = true;
    });
  });

  // Show all layer tags
  const layerTagsList = document.getElementById('layerTagsList');
  Array.from(layerTagsList.children).forEach(tag => {
    tag.style.display = 'block';
  });
}


// Animation Loop to Continuously Render the Scene
function animate() {
  requestAnimationFrame(animate);
  controls.update();  // Update OrbitControls for smooth interactions
  renderer.render(scene, camera);
}

// Start the Animation Loop
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

