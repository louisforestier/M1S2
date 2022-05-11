import React, {useState, useEffect} from "react";
import ReactDOM from "react-dom";

import Application from "./app/app";

const App = (props) => {
  const [events,setEvents] = useState([]);
  useEffect(() => {
      fetch('/api/events')
          .then((res) => res.json())
          .then((eventsResponse) => {
              setEvents(eventsResponse)
          })
  })

    return(
        <div>
            <table>
                <thead>
                <tr>
                    <td>name</td>
                    <td>date</td>
                </tr>
                </thead>
                <tbody>
                {
                    events && events
                        .map((event)=>{
                            return <tr>
                                <td>{event.name}</td>
                                <td>{event.date}</td>
                            </tr>
                        })
                }
                </tbody>
            </table>
        </div>
    )
}


ReactDOM.render(
    <Application/>,
    document.getElementById('main')
)