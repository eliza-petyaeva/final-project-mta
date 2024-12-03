# diploma-mta

The study addressed the attribution problem in e-commerce by creating an MVP web application for evaluating the effectiveness of advertising campaigns. The project involved defining requirements, analyzing data, and implementing the application. The web application provides functionality for data uploading, calculating attribution metrics using eight models, visualizing data, and exporting data for local storage. The supported attribution models include First Touch Attribution, Last Touch Attribution, Linear Attribution, Time Decay Attribution, Position-Based Attribution, Markov Chain Attribution, Shapley Attribution, and Media Mix Model. The MVP enables the analysis and comparison of advertising campaign attribution models for more effective decision making on budget allocation.

Code for the final project: **"Analytical Web-application Development for Effectiveness Evaluation of Advertising Campaigns and Platforms in E-commerce."**

## Instructions for Running the Application:

1. **Set up the environment**  
   Navigate to the `project` folder and activate the environment using the following command:
   ```bash
   poetry install
   ```
2. Start the application
Run the application with:
```bash
poetry run streamlit run main.py
```

3. Load the data
Upload the `data_for_mvp.pickle` file into the application when prompted.


