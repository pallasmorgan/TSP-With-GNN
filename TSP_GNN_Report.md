1. **Title and Introduction**
    
    - TSP With GNN
        
    - The traveling salesman problem is an optimization challege with a "salesman" musr visit a set of cities exactly once, returning to the starting point. The goal is to minimize the total travel cost. We are using Graph Neural Networks to structure the data points(cities), GNNs use message passing mechanisms to aggregate and update node representation based on their neighbors. The next city at a given point is chosen through the decoder method based on these messages passed. 
        
2. **Data Preparation**
    
    -locations = [
    ("Charleston, SC", (32.7765, -79.9311)), ("Mount Pleasant, SC", (32.8325, -79.8577)), ("Sullivan’s Island, SC", (32.6680, -79.9743)), ("Isle of Palms, SC", (32.7524, -79.8431)), ("James Island, SC", (32.7632, -79.9788)), ("West Ashley, SC", (32.8460, -80.0375)), ("Summerville, SC", (33.0181, -80.1756)), ("North Charleston, SC", (32.8781, -80.0043)), ("Hanahan, SC", (32.7870, -80.0077)), ("Moncks Corner, SC", (33.1670, -79.9765)), ("Ladson, SC", (32.9265, -80.0836)), ("Goose Creek, SC", (32.9812, -80.0724)), ("Charleston Neck, SC", (32.9000, -80.1000)), ("West Columbia, SC", (33.9950, -81.0498)), ("Columbia, SC", (34.0007, -81.0348)), ("Lexington, SC", (33.9857, -81.2235)), ("Irmo, SC", (34.0101, -81.0867)), ("St. Andrews, SC", (33.9001, -80.2500)),
("Fort Mill, SC", (34.9871, -80.9698)), ("Rock Hill, SC", (34.9249, -81.0251)), ("Concord, NC", (35.4087, -80.5795)), ("Gastonia, NC", (35.2621, -81.1873)), ("Mount Holly, NC", (35.2790, -81.0608)), ("Belmont, NC", (35.3435, -81.0550)), ("Davidson, NC", (35.4881, -80.8766)), ("Huntersville, NC", (35.4107, -80.8428)),
("Cornelius, NC", (35.4241, -80.8700)),("Conover, NC", (35.4521, -80.8200)),
("Highland, NC", (35.4437, -80.8800)), ("Charlotte, NC", (35.2271, -80.8431))
        
    - A brief explanation of how you computed the distance matrix using the Haversine formula (include the formula and explain each term).
        
3. **Model Architecture**
    
    - A diagram or schematic (e.g., drawn in code or hand‑sketched and embedded as an image) showing your GNN architecture:
        
        - Input node features
            
        - GCN layers (with dimensionalities)
            
        - Linear decoder
            
    - A written explanation of how Graph Convolutional Networks work, including the concept of message passing and aggregation.
        
4. **Implementation Details**
    
    - The fully commented Python code (as shown in the assignment), broken into logical sections (data preparation, model definition, training loop, decoding, visualization).
        
    - Any modifications or extensions you made beyond the starter code.
        
5. **Results and Visualizations**
    
    - Plots of:
        
        - The expert (nearest‑neighbor) tour.
            
        - The learned GNN‑predicted tour.
            
    - For each plot, include a caption describing what is shown.
        
    - If you experimented with different hyperparameters or architectures, include comparison plots or tables.
        
6. **Analysis and Discussion**
    
    - **Haversine Formula:** Explain in your own words why and how it computes great‑circle distances.
        
    - **GNN Theory:** Describe the role of each GNN layer, how node embeddings are updated, and why this is useful for TSP.
        
    - **Training Strategy:** Discuss the supervised imitation‑learning approach (using the nearest‑neighbor heuristic as “expert”), any challenges you faced (e.g., masking visited nodes), and how you addressed them.
        
    - **Performance Comparison:** Compare the total tour lengths (in kilometers) of the expert heuristic versus your GNN model. Discuss any discrepancies and hypothesize why they occur.
        
    - **What You Learned:** Reflect on key takeaways:
        
        - Insights into GNNs and combinatorial optimization.
            
        - Limitations of your approach and possible improvements.
            
        - Broader lessons about applying machine learning to NP‑hard problems.
            
7. **Appendix (Optional)**
    
    - Any additional experiments (e.g., different heuristics, alternative decoders).
        
    - Raw training logs or extended tables.
        

---

**Submission Checklist:**

-  `TSP_GNN_Report.md` (Markdown report with embedded images and analysis)
    
-  All Final Python source files, training, and visualization.
    
-  Any external diagrams or sketches included as image files and referenced in the report.
