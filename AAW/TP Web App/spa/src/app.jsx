import React from "react";
import ReactDOM from "react-dom";

class Application extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            events:[],
            loading:false
        }
    }

    componentDidMount(){
        this.setState({loading:true});
        fetch('/api/event')
        .then((res)=> res.json())
        .then((eventsReponse)=>{
            this.setState({loading:false,events:eventsReponse})
        })
    }

    render(){
        const {loading,events}  = this.state;
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
}