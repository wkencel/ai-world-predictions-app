# AI World Predictions App

The AI World Predictions App is designed to provide AI-driven predictions for world events using a Retrieval-Augmented Generation (RAG) approach. The application scrapes data from various websites, stores it in a vector database (Pinecone), and interacts with APIs such as Polymarket and Kalshi. This data is processed by an OpenAI language model and presented to users through a React-based web application.

## System Architecture

The application is divided into two main components: the client and the server.

### Client

The client is a React application that serves as the user interface for interacting with the AI predictions.

- **Public Directory**: Contains static assets.
  - `index.html`: The main HTML file for the React app.

- **Source Directory**: Contains the core application logic.
  - **components/**: Houses reusable React components.
  - **pages/**: Contains Next.js pages for routing and navigation.
  - **styles/**: Includes CSS/SCSS files for styling the application.
  - **hooks/**: Custom React hooks for managing state and side effects.
  - **utils/**: Utility functions for client-side operations.
  - **context/**: Context API files for global state management.
  - `App.js`: The main component that initializes the React app.
  - `index.js`: The entry point for the React application.

- `package.json`: Lists client-side dependencies and scripts.

### Server

The server handles data processing, API interactions, and serves the client application.

- **API Directory**: Manages external API integrations.
  - **polymarket/**: Handles interactions with the Polymarket API.
  - **kalshi/**: Manages connections to the Kalshi API.

- **Database Directory**: Manages database interactions.
  - **pinecone/**: Contains scripts and configurations for Pinecone vector database.

- **Scraping Directory**: Contains web scraping logic.
  - **scripts/**: Scripts for extracting data from websites.
  - **utils/**: Utility functions to support scraping operations.

- **Services Directory**: Integrates with external services.
  - **openai/**: Manages interactions with OpenAI's language models.

- **Utilities Directory**: General utility functions for server-side logic.

- **Middlewares Directory**: Middleware functions for request processing.

- **Routes Directory**: Defines API endpoints and routing logic.

- **Models Directory**: Data models for structuring and validating data.

- **Controllers Directory**: Business logic and data handling.

- `server.js`: The main server file that initializes and runs the server.

- `package.json`: Lists server-side dependencies and scripts.

### Root

- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Documentation for the project.

## Getting Started

To get started with the project, clone the repository and install the necessary dependencies for both the client and server. Then, run the development servers.

## Collaborators

- Ashley Pean
- Dave Laflam
- Jamie Highsmith
- John Maltese
- Will Kencel
