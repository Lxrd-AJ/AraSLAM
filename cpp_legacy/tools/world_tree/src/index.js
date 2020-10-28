import React from "react";
import ReactDOM from "react-dom";
import Example from "./spikes/Example";
import Worldview, { Cubes, Axes, Text } from "regl-worldview";
import CalibrationBoard from "./elements/CalibrationBoard";
import CameraView from "./elements/CameraView";
import "./styles.css";

function App() {
	return (
		<div className="App" style={{width:"100vw", height: "100vh" }}>
				{/* <h1>World Tree</h1> */}
				{/* <h2>Matchmove software</h2> */}
				
				{/* <Example /> */}

				<Worldview>
					<Axes />
					<CalibrationBoard />
					<CameraView />
				</Worldview>
		</div>
	);
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
