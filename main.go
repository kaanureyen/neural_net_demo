package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// NeuralNetwork defines a simple neural network
type NeuralNetwork struct {
	inputSize, hiddenSize, outputSize int
	hiddenWeights, outputWeights      *mat.Dense
	hiddenBias, outputBias            *mat.Dense

	// pre-allocated matrices
	hiddenInputs, hiddenOutputs *mat.Dense
	finalInputs, finalOutputs   *mat.Dense
	outputErrors, outputDeltas  *mat.Dense
	hiddenErrors, hiddenDeltas  *mat.Dense
	outputWeightsUpdate         *mat.Dense
	hiddenWeightsUpdate         *mat.Dense

	predictBiasHidden  *mat.Dense
	predictBiasOutput  *mat.Dense
	trainOutputBiasSum *mat.Dense
	trainHiddenBiasSum *mat.Dense
	ones               *mat.Dense
}

// NewNeuralNetwork creates a new neural network
func NewNeuralNetwork(input, hidden, output int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	// Initialize weights and biases with random values
	hiddenWeightsData := make([]float64, input*hidden)
	for i := range hiddenWeightsData {
		hiddenWeightsData[i] = rand.Float64() - 0.5
	}
	hiddenWeights := mat.NewDense(input, hidden, hiddenWeightsData)

	outputWeightsData := make([]float64, hidden*output)
	for i := range outputWeightsData {
		outputWeightsData[i] = rand.Float64() - 0.5
	}
	outputWeights := mat.NewDense(hidden, output, outputWeightsData)

	hiddenBiasData := make([]float64, hidden)
	for i := range hiddenBiasData {
		hiddenBiasData[i] = rand.Float64() - 0.5
	}
	hiddenBias := mat.NewDense(1, hidden, hiddenBiasData)

	outputBiasData := make([]float64, output)
	for i := range outputBiasData {
		outputBiasData[i] = rand.Float64() - 0.5
	}
	outputBias := mat.NewDense(1, output, outputBiasData)

	// Create a 1x4 matrix of ones for summing columns
	onesData := make([]float64, 4)
	for i := range onesData {
		onesData[i] = 1.0
	}
	ones := mat.NewDense(1, 4, onesData)

	nn := &NeuralNetwork{
		inputSize:     input,
		hiddenSize:    hidden,
		outputSize:    output,
		hiddenWeights: hiddenWeights,
		outputWeights: outputWeights,
		hiddenBias:    hiddenBias,
		outputBias:    outputBias,

		// pre-allocated matrices
		hiddenInputs:        mat.NewDense(4, hidden, nil),
		hiddenOutputs:       mat.NewDense(4, hidden, nil),
		finalInputs:         mat.NewDense(4, output, nil),
		finalOutputs:        mat.NewDense(4, output, nil),
		outputErrors:        mat.NewDense(4, output, nil),
		outputDeltas:        mat.NewDense(4, output, nil),
		hiddenErrors:        mat.NewDense(4, hidden, nil),
		hiddenDeltas:        mat.NewDense(4, hidden, nil),
		outputWeightsUpdate: mat.NewDense(hidden, output, nil),
		hiddenWeightsUpdate: mat.NewDense(input, hidden, nil),

		predictBiasHidden:  mat.NewDense(4, hidden, nil),
		predictBiasOutput:  mat.NewDense(4, output, nil),
		trainOutputBiasSum: mat.NewDense(1, output, nil),
		trainHiddenBiasSum: mat.NewDense(1, hidden, nil),
		ones:               ones,
	}

	return nn
}

// sigmoid activation function
func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// sigmoid derivative
func sigmoidDerivative(r, c int, z float64) float64 {
	return z * (1.0 - z)
}

// Predict performs the forward pass
func (nn *NeuralNetwork) Predict(inputs mat.Matrix) (*mat.Dense, *mat.Dense) {
	nn.hiddenInputs.Reset()
	nn.hiddenOutputs.Reset()
	nn.finalInputs.Reset()
	nn.finalOutputs.Reset()

	// hidden layer
	nn.hiddenInputs.Mul(inputs, nn.hiddenWeights)
	nn.predictBiasHidden.Apply(func(r, c int, v float64) float64 { return nn.hiddenBias.At(0, c) }, nn.predictBiasHidden)
	nn.hiddenInputs.Add(nn.hiddenInputs, nn.predictBiasHidden)
	nn.hiddenOutputs.Apply(sigmoid, nn.hiddenInputs)

	// output layer
	nn.finalInputs.Mul(nn.hiddenOutputs, nn.outputWeights)
	nn.predictBiasOutput.Apply(func(r, c int, v float64) float64 { return nn.outputBias.At(0, c) }, nn.predictBiasOutput)
	nn.finalInputs.Add(nn.finalInputs, nn.predictBiasOutput)
	nn.finalOutputs.Apply(sigmoid, nn.finalInputs)

	return nn.hiddenOutputs, nn.finalOutputs
}

// Train performs the backpropagation
func (nn *NeuralNetwork) Train(inputs, targets mat.Matrix, learningRate float64) {
	nn.outputErrors.Reset()
	nn.outputDeltas.Reset()
	nn.hiddenErrors.Reset()
	nn.hiddenDeltas.Reset()
	nn.outputWeightsUpdate.Reset()
	nn.hiddenWeightsUpdate.Reset()

	hiddenOutputs, finalOutputs := nn.Predict(inputs)

	// output errors
	nn.outputErrors.Sub(targets, finalOutputs)

	// output deltas
	nn.outputDeltas.Apply(sigmoidDerivative, finalOutputs)
	nn.outputDeltas.MulElem(nn.outputErrors, nn.outputDeltas)

	// hidden errors
	nn.hiddenErrors.Mul(nn.outputDeltas, nn.outputWeights.T())

	// hidden deltas
	nn.hiddenDeltas.Apply(sigmoidDerivative, hiddenOutputs)
	nn.hiddenDeltas.MulElem(nn.hiddenErrors, nn.hiddenDeltas)

	// update output weights and biases
	nn.outputWeightsUpdate.Mul(hiddenOutputs.T(), nn.outputDeltas)
	nn.outputWeightsUpdate.Scale(learningRate, nn.outputWeightsUpdate)
	nn.outputWeights.Add(nn.outputWeights, nn.outputWeightsUpdate)

	// Update output bias
	nn.trainOutputBiasSum.Set(0, 0, mat.Sum(nn.outputDeltas))
	nn.trainOutputBiasSum.Scale(learningRate, nn.trainOutputBiasSum)
	nn.outputBias.Add(nn.outputBias, nn.trainOutputBiasSum)

	// update hidden weights and biases
	nn.hiddenWeightsUpdate.Mul(inputs.T(), nn.hiddenDeltas)
	nn.hiddenWeightsUpdate.Scale(learningRate, nn.hiddenWeightsUpdate)
	nn.hiddenWeights.Add(nn.hiddenWeights, nn.hiddenWeightsUpdate)

	// Update hidden bias
	nn.trainHiddenBiasSum.Mul(nn.ones, nn.hiddenDeltas)
	nn.trainHiddenBiasSum.Scale(learningRate, nn.trainHiddenBiasSum)
	nn.hiddenBias.Add(nn.hiddenBias, nn.trainHiddenBiasSum)
}

func main() {
	// XOR training data
	inputs := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	targets := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	// Create a new neural network
	nn := NewNeuralNetwork(2, 2, 1)

	// Train the neural network
	epochs := 1000000
	learningRate := 0.5
	for i := 0; i < epochs; i++ {
		nn.Train(inputs, targets, learningRate)
	}

	// Test the neural network
	fmt.Println("XOR function approximation:")
	_, prediction := nn.Predict(inputs)
	fmt.Printf("Input: %v, Prediction: %v\n", mat.Formatted(inputs), mat.Formatted(prediction))
}
