1. TSP With GNN
    
        
    - The traveling salesman problem is an optimization challege with a "salesman" musr visit a set of cities exactly once, returning to the starting point. The goal is to minimize the total travel cost. We are using Graph Neural Networks to structure the data points(cities), GNNs use message passing mechanisms to aggregate and update node representation based on their neighbors. The next city at a given point is chosen through the decoder method based on these messages passed. 
        
2. **Data Preparation**
    
    -locations = [
    ("Charleston, SC", (32.7765, -79.9311)), ("Mount Pleasant, SC", (32.8325, -79.8577)), ("Sullivan’s Island, SC", (32.6680, -79.9743)), ("Isle of Palms, SC", (32.7524, -79.8431)), ("James Island, SC", (32.7632, -79.9788)), ("West Ashley, SC", (32.8460, -80.0375)), ("Summerville, SC", (33.0181, -80.1756)), ("North Charleston, SC", (32.8781, -80.0043)), ("Hanahan, SC", (32.7870, -80.0077)), ("Moncks Corner, SC", (33.1670, -79.9765)), ("Ladson, SC", (32.9265, -80.0836)), ("Goose Creek, SC", (32.9812, -80.0724)), ("Charleston Neck, SC", (32.9000, -80.1000)), ("West Columbia, SC", (33.9950, -81.0498)), ("Columbia, SC", (34.0007, -81.0348)), ("Lexington, SC", (33.9857, -81.2235)), ("Irmo, SC", (34.0101, -81.0867)), ("St. Andrews, SC", (33.9001, -80.2500)),
("Fort Mill, SC", (34.9871, -80.9698)), ("Rock Hill, SC", (34.9249, -81.0251)), ("Concord, NC", (35.4087, -80.5795)), ("Gastonia, NC", (35.2621, -81.1873)), ("Mount Holly, NC", (35.2790, -81.0608)), ("Belmont, NC", (35.3435, -81.0550)), ("Davidson, NC", (35.4881, -80.8766)), ("Huntersville, NC", (35.4107, -80.8428)),
("Cornelius, NC", (35.4241, -80.8700)),("Conover, NC", (35.4521, -80.8200)),
("Highland, NC", (35.4437, -80.8800)), ("Charlotte, NC", (35.2271, -80.8431))
        
    -  **Distance Matrix with Haversine Formula:**
  ```python
  from math import radians, cos, sin, asin, sqrt
  def haversine(coord1, coord2):
      lat1, lon1 = radians(coord1[0]), radians(coord1[1])
      lat2, lon2 = radians(coord2[0]), radians(coord2[1])
      dlon = lon2 - lon1
      dlat = lat2 - lat1
      a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
      c = 2 * asin(sqrt(a))
      r = 6371  # Radius of Earth in km
      return c * r
  ```
  - This formula calculates great-circle distances between points on a sphere given their longitudes and latitudes.

        
3. **Model Architecture**
    
    - **Architecture Diagram (insert image):**
    - Input: Node features (coordinates)
    - 3-layer GCN with ReLU and Dropout
    - Linear decoder for output logits

- **Explanation:**
    - GCNs update each node’s representation by aggregating its neighbors’ features.
    - After passing through GCN layers, node embeddings encode spatial and relational information.
    - The decoder (pointer mechanism) selects the next node during tour construction.

        
4. **Implementation Details**
    - **Code Structure:**
    - `prepare_data()` – Haversine matrix, edge index
    - `TSP_GNN` – 3-layer GCN model
    - `decode()` – greedy decoding using pointer network
    - `two_opt()` – iterative edge-swap optimization
    - `train()` – supervised imitation learning loop
    - `visualize()` – matplotlib-based visualization
    - The fully commented Python code (as shown in the assignment), broken into logical sections (data preparation, model definition, training loop, decoding, visualization).
        
    - Any modifications or extensions you made beyond the starter code.
    - - **Random Restart:** The algorithm is now run 10 times with a random starting node. This can help escape local minima by diversifying the starting points and potentially finding different tour configurations.

    - **Two-Opt Heuristic:** This swaps two edges in the tour and checks if it improves the total distance. If it does, the swap is applied. It iteratively tries to improve the tour by eliminating "crossed" edges.

    - **Parameters**
        - `use_random_restart` (default True): Enables random restarts to avoid local minima by varying the starting point.
        - `use_two_opt` (default True): Enables the Two-Opt heuristic to attempt improving the tour after it’s generated.

    - **Training Loop Improvements**
        - Additional heuristics that help escape local minima in the neural network approach using a few techniques. These assist in diversifying the search process and reduce the likelihood of the algorithm getting stuck in suboptimal tours.
        - State Representation Encoding
        - Dynamic Node Selection (Greedy Decoding)
        - Step-wise Supervision with Expert Targets
        - Score Computation Per Node
        - Modular State Representation

    - **Decoding Enhancements**
        - Explore multiple possible tours in parallel
        - Dynamic state representation that updates as the tour is constructed
        - Integration of the edge weights in the decision process
        - Temperature scaling on the scores to control exploration vs. exploitation

        
5. **Results and Visualizations**
    
    - Plots of:
        
        -![image](https://github.com/user-attachments/assets/066e8aca-67ea-46d4-b8d5-19618ea988ce)

            
     **Comparisons:**
    - Tour Lengths: e.g., NN = 895km, GNN = 879km
        
6. **Analysis and Discussion**
    
    - **Haversine Formula:** Explain in your own words why and how it computes great‑circle distances.
        
    - **GNN Theory:**
    - Graph Neural Networks are built to handle graph-structured data, making them a natural fit for the Traveling Salesman Problem, where each city is a node and distances are edges.
    - GNNs work by passing messages between nodes—each node updates its representation by aggregating information from its neighbors.
    - After a few layers of this, each node has a learned embedding that reflects both its own features and its position in the graph. This is exactly what TSP needs: city embeddings that capture spatial relationships.

   - **Forward Method:**
    - The forward method is where the model processes the graph.
    - It takes in node features, edge connections, and edge attributes, and runs them through three `GCNConv` layers with ReLU activations in between.
    - Each layer allows the nodes to learn from a wider neighborhood. After the final layer, a linear layer outputs a score for each node, representing how strong of a candidate it is to be the next city in the tour.
    - The method returns both the scores and final embeddings—compact, clean, and effective for route prediction.

- **Model Improvements and Justification:**
    - Dropout after each activation to reduce overfitting and encourage more robust learning.
    - Added a third GCN layer to give the model deeper capacity—this helps each node learn from a larger portion of the graph.
    - While edge features are not used yet, future improvements could include edge-aware architectures like Graph Attention Networks or EdgeConv, allowing the model to reason more directly about inter-city distances.

- **Training Strategy:**
    - Discuss the supervised imitation‑learning approach (using the nearest‑neighbor heuristic as “expert”), any challenges you faced (e.g., masking visited nodes), and how you addressed them.
    - Pre-established target value outside the loop
    - Updated target to have a parameter `device = device`
    - Added a checker to make sure tensor and models are on the same device before entering the loop
    - Changed visited to a `torch.BoolTensor`, set the device to `next(model.parameters()).device`, used `masked_fill(visited, -inf)`, and removed current dependency

- **Visualization Enhancements:**
    ```python
    lats, lons = zip(*coords)
    plt.plot(lons, lats, linestyle='-', marker='o')
    for i, (lat, lon) in enumerate(coords):
        plt.text(lon + 0.002, lat + 0.002, str(i))
    plt.tight_layout()

- **Performance Comparison:**
   - GNN outperformed NN heuristic in 7/10 runs
    - Random restarts and Two-Opt helped reduce final distance

- - **Insights:**
    - GNNs can generalize spatial dependencies in TSP
    - Hybrid classical-quantum models offer promising future directions
    - Challenges: maintaining efficiency, encoding edge weights, and multi-agent routing

        

---

**Submission Checklist:**

-  `TSP_GNN_Report.md` (Markdown report with embedded images and analysis)
    
-  All Final Python source files, training, and visualization.
    
-  Any external diagrams or sketches included as image files and referenced in the report.
