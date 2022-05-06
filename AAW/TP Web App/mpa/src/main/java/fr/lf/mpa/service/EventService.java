package fr.lf.mpa.service;

import fr.lf.mpa.context.Context;
import fr.lf.mpa.form.EventForm;
import fr.lf.mpa.model.Event;
import fr.lf.mpa.model.EventRecord;
import fr.lf.mpa.model.Person;
import fr.lf.mpa.repository.EventRepository;
import fr.lf.mpa.repository.PersonRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class EventService {

    @Autowired
    private Context context;

    @Autowired
    private PersonRepository personRepository;

    @Autowired
    EventRepository eventRepository;

    @Value("false")
    private Boolean withContext;

    public Context getContext() {
        return context;
    }

    public Boolean withContext() {
        return withContext;
    }

    public void setContext(Context context) {
        this.context = context;
    }

    public void delEvent(UUID id) {
        if (withContext){
            List<EventRecord> eventsFiltered = context.getEvents()
                    .stream()
                    .filter(e -> !e.getId().equals(id))
                    .collect(Collectors.toList());
            context.updateEvents(eventsFiltered);
        } else {
            eventRepository.deleteById(id);
        }
    }

    public void addEvent(EventForm eventForm){
        if (withContext) {
            context.getEvents().add(new EventRecord(UUID.randomUUID(), eventForm.getName(), eventForm.getDate()));
        } else {
            Event event = new Event();
            event.setName(eventForm.getName())
                    .setDate(eventForm.getDate());
            eventRepository.save(event);
        }
    }

    public List<Event> getEvents(){
        List<Event> events = new ArrayList<>();
        eventRepository.findAll().forEach(events::add);
        return events;
    }

    public List<Person> getPersons(){
        List<Person> persons = new ArrayList<>();
        personRepository.findAll().forEach(persons::add);
        return persons;
    }
}
