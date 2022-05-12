import React, {useState} from "react";
/* si on laisse props dans les paramètres du composant
props:{
    loadEvents:Function
}
*/
const AddEvent = ({loadEvents})=> {
    const [name, setName] = useState("Saisir un nom");
    const [date, setDate] = useState("2022-01-01");

    const valider = (e) => {
        e.preventDefault();
        let body = JSON.stringify({name: name,date:date});
        fetch('/api/events', {
            method: "POST",
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: body
        })
            .then((res) => res.json())
            .then((eventsReponse) => {
                loadEvents();
            })
    }

    return(
        <form onSubmit={valider}>
            <input type={"text"} value={name} onChange={(e) => setName(e.currentTarget.value)}/>
            <input type={"date"} value={date} onChange={(e) => setDate(e.currentTarget.value)}/>
            <button>Valider</button>
        </form>
    )
}

export default AddEvent;