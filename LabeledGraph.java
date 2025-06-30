package program;

import mklab.JGNN.core.Matrix;

public class LabeledGraph {
	// Instance fields
	public Matrix adjacency;
	public Matrix nodeFeatures;
	public Matrix edgeFeatures; // set to null if not using
	public int label;

	// Constructor
	public LabeledGraph(Matrix adjacency, Matrix nodeFeatures, Matrix edgeFeatures, int label) {
		this.adjacency = adjacency;
		this.nodeFeatures = nodeFeatures;
		this.edgeFeatures = edgeFeatures;
		this.label = label;
	} // END constructor
} // END "LabeledClass" class
