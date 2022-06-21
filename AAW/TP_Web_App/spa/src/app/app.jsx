import React from "react";
import AddEvent from "../component/addEvent";

class Application extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            events: [],
/* si on veut mettre un gros objet plutot que de mettre les champs à plat
            event:{
                name:"Saisir un nom",
                date:"01/01/1900"
            },
*/
            name:"Saisir un nom",
            date:"2022-01-01",
            loading: false
        }
    }

    componentDidMount() {
        this.loadEvents();
    }

    loadEvents = () => {
        this.setState({loading: true});
        fetch('/api/events')
            .then((res) => res.json())
            .then((eventsReponse) => {
                this.setState({loading: false, events: eventsReponse})
            })
    }

    valider = (e) => {
        e.preventDefault();
        this.setState({loading: true});
        let body = JSON.stringify({name: this.state.name,date:this.state.date});
        console.log(body);
        fetch('/api/events', {
            method: "POST",
            body: body,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        })
            .then((res) => res.json())
            .then((eventsReponse) => {
                this.setState({loading: false, events: eventsReponse})
            })
    }

    suppr(event){
        fetch('/api/events/'+event.id, {
            method: "DELETE"
        })
            .then((res)=>res.json())
            .then((eventsReponse) => {
                console.log(eventsReponse);
                this.setState({loading: false, events: eventsReponse})
            })
    }

    render() {
        const {loading, events} = this.state;
        return (
            <div>
                <table>
                    <thead>
                    <tr>
                        <td>id</td>
                        <td>name</td>
                        <td>date</td>
                    </tr>
                    </thead>
                    <tbody>
                    {
                        events && events
                            .map((event) => {
                                return <tr>
                                    <td>{event.id}</td>
                                    <td>{event.name}</td>
                                    <td>{event.date}</td>
                                    <td>
                                        <button onClick={(e) => {
                                        this.suppr(event)
                                    }}>x</button>
                                    </td>
                                </tr>
                            })
                    }
                    </tbody>
                </table>

                <AddEvent loadEvents={this.loadEvents}/>


{/*
                <form onSubmit={this.valider}>
                    <input type={"text"} value={this.state.name} onChange={(e) => this.setState({name:e.currentTarget.value})}/>
                    <input type={"date"} value={this.state.date} onChange={(e) => this.setState({date:e.currentTarget.value})}/>
                    <button>Valider</button>
                </form>
*/}

            </div>
        )
    }
}


export default Application;