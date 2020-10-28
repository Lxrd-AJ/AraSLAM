import React from "react";
import Worldview, { Cubes, Axes, Text } from "regl-worldview";

export default function CalibrationBoard() {
    const name = "CalibrationBoard module";

    return (
        <Cubes>
            {[
                {
                    pose: {
                        orientation: {x:0, y:0, z:0, w:1 },
                        position: { x:-25, y: -25, z:0 },
                    },
                    scale: { x:50, y:50, z:1 },
                    color: { r:1, g:1, b:1, a:1 }
                }
            ]}
        </Cubes>        
    );
}