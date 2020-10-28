import React from "react";
import _, { Cones } from "regl-worldview";
import jsyaml from "js-yaml";
// const fs = window.require('fs');

// export default function CameraView(props){
    // props.markers = [];
    // const readFile = async (name) => {
    //     const contents = await fetch(name);
    //     const text = await contents.text();
    //     return text;
    // }

    // const yaml = readFile("./file.yaml").then((txt) => {            
    //     var doc = jsyaml.safeLoadAll(txt);
    //     const name = doc.shift();
    //     doc.forEach((obj) => {
    //         const rot = obj["rotation"];
    //         const tran = obj["translation"];
    //         console.log(rot);
    //         console.log(tran)
    //         props.markers.push({
    //             pose: {
    //                 orientation: {x:rot[0], y:rot[1], z:rot[2], w:1 },
    //                 position: {x:tran[0], y:tran[1], z:tran[2]}
    //             },
    //             scale: {x:10, y:10, z:10},
    //             color: {r:1, g:0, b:1, a:0.8},
    //         });
    //     });
    // });


    // return (
    //     <Cones>{props.markers}</Cones>
    // )
// }

class CameraView extends React.Component {
    constructor(props){
        super(props);
        this.state = { markers: [] };
    }

    componentDidMount(){
        const readFile = async (name) => {
            const contents = await fetch(name);
            const text = await contents.text();
            return text;
        }
    
        readFile("./file.yaml").then((txt) => {            
            var doc = jsyaml.safeLoadAll(txt);
            doc.shift();
            doc.forEach((obj) => {
                const rot = obj["rotation"][0];                
                const tran = obj["translation"][0];
                let marker = [{
                    pose: {
                        orientation: { x: rot[0], y:rot[1], z:rot[2], w:1 },
                        position: {x:tran[0], y:tran[1], z:tran[2]}
                    },
                    scale: {x:5, y:5, z:5},
                    color: {r:1, g:0, b:1, a:0.8},
                }];
                
                let new_markers = this.state.markers.concat(marker);
                console.log(new_markers)
                this.setState((state,props) => ({
                    markers: new_markers
                }))
            });
        });        
    }

    render(){
        return (
            <Cones>{this.state.markers}</Cones>
        )
    }
}

export default CameraView;