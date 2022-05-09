import React from "react";
import ReactDOM from "react-dom";

import Application from "./app";

let Comp = () => {
    return <div>Bonjour !</div>;
}

ReactDOM.render(
    <Comp/>,
    document.getElementById('main')
)

ReactDom.render(
    <Application/>,
    document.getElementById('main')
)