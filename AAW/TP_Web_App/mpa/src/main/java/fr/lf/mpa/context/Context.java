package fr.lf.mpa.context;

import fr.lf.mpa.model.EventRecord;
import fr.lf.mpa.model.PersonRecord;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Component
public class Context {

    private List<EventRecord> events = new ArrayList<>();
    private List<PersonRecord> persons = new ArrayList<>();

    public Context(){
        events.add(new EventRecord(UUID.randomUUID(),"Mud Day", LocalDate.now().toString()));
        events.add(new EventRecord(UUID.randomUUID(),"Vide grenier", LocalDate.now().toString()));

        persons.add(new PersonRecord(UUID.randomUUID(),"Bill", "Gates",events.get(0)));
        persons.add(new PersonRecord(UUID.randomUUID(),"Steve", "Jobs",events.get(1)));
    }

    public List<EventRecord> getEvents() {
        return events;
    }

    public void setEvents(List<EventRecord> events) {
        this.events = events;
    }

    public List<PersonRecord> getPersons() {
        return persons;
    }

    public void setPersons(List<PersonRecord> persons) {
        this.persons = persons;
    }

    public void updatePersons(List<PersonRecord> personsFiltered) {
        persons.clear();
        persons.addAll(personsFiltered);
    }

    public void updateEvents(List<EventRecord> eventsFiltered) {
        events.clear();
        events.addAll(eventsFiltered);
    }
}
