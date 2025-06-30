package program;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IBond.Stereo;
import org.openscience.cdk.ringsearch.RingSearch;
import org.openscience.cdk.interfaces.IAtomType.Hybridization;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.graph.Cycles;
import org.openscience.cdk.aromaticity.ElectronDonation;

// I needed to use CDK's jar downloaded feom their GitHub to get it to work
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.matrix.DenseMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SmilesToGraphExample {

	public static void main(String[] args) throws Exception {
		// Initialize the SMILES parser
		SmilesParser sp = new SmilesParser(DefaultChemObjectBuilder.getInstance());

		// Example SMILES for benzene with its experimentally determined binary classifying label of hit or no-hit ("1" or "0" respectively)
		String fullData = "c1ccccc1|0";
		String[] parts = fullData.split("\\|");
		String smiles = parts[0];
		int label = Integer.parseInt(parts[1]);

		// Parse SMILES to molecule
		IAtomContainer molecule = sp.parseSmiles(smiles);
		AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);

		// Determine aromaticity - a special chemistry property for certain ring molecules (some rings are aromatic, some are not)
		// This ensures all atoms and bonds have proper aromaticity information!
		Aromaticity aromaticity = new Aromaticity(ElectronDonation.daylight(), Cycles.all());
		aromaticity.apply(molecule);
        
		// Precompute ring information
		RingSearch ringSearch = new RingSearch(molecule);

		// Output basic information
		System.out.println("Number of atoms: " + molecule.getAtomCount());
		System.out.println("Number of bonds: " + molecule.getBondCount());

		// Build atom index mapping
		int n = molecule.getAtomCount();
		Map<IAtom, Integer> atomIndex = new HashMap<>();
		for (int i = 0; i < n; i++) {
			atomIndex.put(molecule.getAtom(i), i);
		} // END 'for'

		// Build adjacency matrix
		Matrix adjacency = new DenseMatrix(n, n);
		for (IBond bond : molecule.bonds()) {
			int i = atomIndex.get(bond.getAtom(0));
			int j = atomIndex.get(bond.getAtom(1));
			adjacency.put(i, j, bond.getOrder().numeric());
			adjacency.put(j, i, bond.getOrder().numeric()); // undirected
		} // END 'for'

		// Enhanced node features matrix
		List<double[]> nodeFeaturesList = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			IAtom atom = molecule.getAtom(i);
			List<Double> features = new ArrayList<>();

			// Core atomic properties
			features.add((double) atom.getAtomicNumber());
			features.add((double) atom.getFormalCharge());
			features.add(atom.isAromatic() ? 1.0 : 0.0);
			features.add((double) atom.getImplicitHydrogenCount());
			features.add(atom.getExactMass() != null ? atom.getExactMass() : 0.0);
			features.add((double) molecule.getConnectedAtomsCount(atom));

            
			// Hybridization (one-hot encoded)
			Hybridization[] hybridTypes = {
                		Hybridization.SP3, Hybridization.SP2, 
                		Hybridization.SP1, Hybridization.PLANAR3
			};
			for (Hybridization hybrid : hybridTypes) {
				features.add(atom.getHybridization() == hybrid ? 1.0 : 0.0);
			} // END 'for'
            
			// Ring membership
			features.add(ringSearch.cyclic(atom) ? 1.0 : 0.0);
            
			// Valency
			features.add(AtomContainerManipulator.getBondOrderSum(molecule, atom));

			// Convert to array and add
			nodeFeaturesList.add(features.stream().mapToDouble(Double::doubleValue).toArray());
		} // END 'for'

		// !!! JGNN WORK STARTS HERE !!!
        	// Build node feature matrix
        	int numNodeFeatures = nodeFeaturesList.get(0).length;
        	Matrix nodeFeatures = new DenseMatrix(n, numNodeFeatures);
        	for (int i = 0; i < n; i++) {
			double[] features = nodeFeaturesList.get(i);
			for (int j = 0; j < numNodeFeatures; j++) {
				nodeFeatures.put(i, j, features[j]);
			} // END 'for'
        	} // END 'for'

		// Enhanced edge features
		List<double[]> edgeFeaturesList = new ArrayList<>();
		for (IBond bond : molecule.bonds()) {
			List<Double> features = new ArrayList<>();
            
			// Core bond properties
			features.add((double)bond.getOrder().numeric());
			features.add(bond.isAromatic() ? 1.0 : 0.0);
			features.add(ringSearch.cyclic(bond) ? 1.0 : 0.0);
            
			// Stereo configuration
			Stereo stereo = bond.getStereo();
			features.add(stereo == Stereo.UP ? 1.0 : 0.0);
			features.add(stereo == Stereo.DOWN ? 1.0 : 0.0);
			features.add(stereo == Stereo.UP_OR_DOWN ? 1.0 : 0.0);
            
			edgeFeaturesList.add(features.stream().mapToDouble(Double::doubleValue).toArray());
		} // END 'for'
        
		// Build edge feature matrix
		int numBonds = edgeFeaturesList.size();
		int numEdgeFeatures = !edgeFeaturesList.isEmpty() ? edgeFeaturesList.get(0).length : 0;
		Matrix edgeFeatures = new DenseMatrix(numBonds, numEdgeFeatures);
		for (int i = 0; i < numBonds; i++) {
			double[] features = edgeFeaturesList.get(i);
			for (int j = 0; j < numEdgeFeatures; j++) {
				edgeFeatures.put(i, j, features[j]);
			} // END 'for'
		} // END 'for'

		// *** PROGRESS SO FAR: Output matrices ***
		System.out.println("Enhanced Adjacency Matrix:\n" + adjacency);
		System.out.println("Enhanced Node Features:\n" + nodeFeatures);
		System.out.println("Edge Features:\n" + edgeFeatures);

		// Storing matrices into ArrayList for feeding to graph neural net
		// "LabeledGraph" is a custom class to conveniently package the three matrices and their corresponding binary classifier value ("0" or "1") per each molecule from the complete dataset, into one "item" that can be conveniently fed to JGNN's actual graph neural net deep learning model
		List<LabeledGraph> dataset = new ArrayList<>();
		LabeledGraph graph = new LabeledGraph(adjacency, nodeFeatures, edgeFeatures, label);
		dataset.add(graph);

		// Cycle through each object named "graph" in "dataset" to feed it to JGNN

	} // END 'main'
} // END class 'SmilesToGraphExample'
