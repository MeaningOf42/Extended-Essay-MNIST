const picWidth  = 28;
const picHeight = 28;
const pixelSize = 13;
const picIndent = 12;

let modeH1;
let predictionP;
let digitH1;
let accuracyP;
let mode = false;
let modes = (x) => x?"Test":"Training";

let pixelFuncs = [
  (x) => x?1:-1, // Should be on, punnishment if not
  (x) => x?1:0,  // Should be on, no punnishment if not
  (x) => 0,
  (x) => x?0:1,  // Should be
  (x) => x?-1:1
];

let currentMatrix = nullMatrix(picWidth,picHeight);
let model;
for (var data = []; data.push([]) < 10;);

let mnist;
let imageNum = 0;

let totalTested = 0;
let totalRight = 0;

function nullMatrix(width, height) {
  main = [];
  for (let y = 0; y<height; y++) {
    let sub = [];
    for (let x = 0; x<width; x++) {
      sub.push(0);
    }
    main.push(sub);
  }
  return main;
}

function drawMatrix(squareSize, colors, leftX, topY, matrix) {
  for (let xi = 0; xi < matrix[0].length; xi++) {
    for (let yi = 0; yi < matrix.length; yi++) {
      x = leftX + squareSize*xi;
      y = topY  + squareSize*yi;
      fill(colors[matrix[yi][xi]]);
      noStroke();

      rect(x, y, squareSize, squareSize);
    }
  }
}

function drawArray(squareSize, leftX, topY, arr, width, height) {
  for (let xi = 0; xi < width; xi++) {
    for (let yi = 0; yi < height; yi++) {
      let i = xi + (yi*width);
      let x = leftX + squareSize*xi;
      let y = topY  + squareSize*yi;
      fill(color(arr[i]*2));
      noStroke();

      rect(x, y, squareSize, squareSize);
    }
  }
}

function arrToMatrix(arr, width, height, threshhold) {
  let simpleImage = [];
  for (let y = 0; y<height; y++) {
    simpleImage.push([]);
    for (let x=0; x<width; x++) {
      simpleImage[y].push(arr[x+(y*width)] > threshhold);
    }
  }
  return simpleImage;
}

function createModel(layerSizes) {
  let model = tf.sequential();
  console.log("created empty model");
  console.log("Created input layer");
  for (let i = 0; i < layerSizes.length; i++) {
    console.log("adding hidden/out layer: ", i);
    if (i==0) {
      model.add(tf.layers.dense({units:layerSizes[i], inputShape:picWidth*picHeight, activation:"sigmoid"}));
    }
    else {
      model.add(tf.layers.dense({units:layerSizes[i],activation:"sigmoid"}));
    }
  }
  console.log("Created hidden and output layers");
  model.compile({optimizer:"sgd", loss:"binaryCrossentropy", lr:0.01});
  console.log("compiled model");
  return model;
}

async function trainedModel() {
  console.log("loadMNIST");
  await loadMNIST((data) => {
    mnist = data;
  });
  console.log("loadMNIST done");
  console.log("creating x, y tensors");
  let xs_arr = [];
  let small = 60000; // i<mnist.train_images.length
  for (let i = 0; i<small; i++) {
    xs_arr.push(Array.from(mnist.train_images[i]));
  }
  let xs = tf.tensor2d(xs_arr);
  console.log("created x tensor");
  let ys_arr = [];
  for (let i = 0; i<small; i++) {
    let outVec = []
    for (let j=0; j<10; j++) {
      if (mnist.train_labels[i]==j) {
        outVec.push(1);
      }
      else {
        outVec.push(0);
      }
    }
    ys_arr.push(outVec);
  }
  let ys = tf.tensor2d(ys_arr);
  console.log("created y tensors ");
  console.log("creating model")
  let model = createModel([10]);
  console.log("model created")
  console.log("fitting model");
  await model.fit(xs, ys, {
    batchSize: 1,
    epochs: 1
  });
  console.log("model fitted");

  mode = true;
  return model
}

function setup() {
  console.log("woot");
  trainedModel().then((out)=>{
    model=out;
  });
  modeH1      = document.getElementById("modeH1");
  predictionP = document.getElementById("predictionP");
  digitH1     = document.getElementById("digitH1");
  accuracyP   = document.getElementById("accuracyP");
  progressP   = document.getElementById("progressP");
  modeH1.innerHTML = modes(mode);

  var canvas = createCanvas(1050, 400);
  canvas.parent('sketch');
  background(255, 0, 200);
}

function draw() {
  if (mode) {
    drawArray(pixelSize, picIndent, picIndent, mnist.test_images[imageNum], 28,28);
    digitH1.innerHTML = "Digit: " + String(mnist.test_labels[imageNum]);

    let xs = tf.tensor2d([Array.from(mnist.test_images[imageNum])]);
    let prediction = model.predict(xs).dataSync();

    let bestDigitScore = -100;
    let bestDigit = -1;
    for (let i = 0; i<10; i++) {
      if (prediction[i]>bestDigitScore) {
        bestDigitScore = prediction[i];
        bestDigit = i;
      }
    }


    if (bestDigit == mnist.test_labels[imageNum]) {totalRight++;}
    totalTested++;
    accuracyP.innerHTML = "Accuracy: " + String(totalRight/totalTested);
    predictionP.innerHTML = "Prediction: " + String(bestDigit);
    imageNum++;
    progressP.innerHTML = "Image: " + String(imageNum) + " of: " + String(mnist.test_labels.length);
    if (imageNum >= mnist.test_labels.length) {
      console.log("Done.")
      noLoop();
    }
  }
  else {
  }
}
