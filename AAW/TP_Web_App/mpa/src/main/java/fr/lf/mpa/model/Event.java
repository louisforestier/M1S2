package fr.lf.mpa.model;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import java.util.UUID;

@Entity
public class Event {

    @Id
    @GeneratedValue
    private UUID id;
    private String name;
    private String date;

    public Event(UUID id, String name, String date) {
        this.id = id;
        this.name = name;
        this.date = date;
    }

    public Event() {
    }

    public UUID getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public Event setName(String name) {
        this.name = name;
        return this;
    }

    public String getDate() {
        return date;
    }

    public Event setDate(String date) {
        this.date = date;
        return this;
    }
}
