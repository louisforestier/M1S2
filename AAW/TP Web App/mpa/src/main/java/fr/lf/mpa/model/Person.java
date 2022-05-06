package fr.lf.mpa.model;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.ManyToOne;
import java.util.UUID;

@Entity
public class Person {

    @Id
    @GeneratedValue
    private UUID id;
    private String firstName;
    private String lastName;

    @ManyToOne
    private Event event;

    public Person() {
    }

    public UUID getId() {
        return id;
    }

    public String getFirstName() {
        return firstName;
    }

    public Person setFirstName(String firstName) {
        this.firstName = firstName;
        return this;
    }

    public String getLastName() {
        return lastName;
    }

    public Person setLastName(String lastName) {
        this.lastName = lastName;
        return this;
    }

    public Event getEvent() {
        return event;
    }

    public Person setEvent(Event event) {
        this.event = event;
        return this;
    }
}
