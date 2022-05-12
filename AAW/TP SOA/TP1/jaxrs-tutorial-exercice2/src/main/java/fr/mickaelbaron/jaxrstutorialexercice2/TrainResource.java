package fr.mickaelbaron.jaxrstutorialexercice2;

import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.List;
import java.util.Optional;

@Path("trains")
@Produces({MediaType.APPLICATION_XML,MediaType.APPLICATION_JSON})
public class TrainResource {

    public TrainResource(){}

    @GET
    public List<Train> getTrains(){
        System.out.println("getTrains");
        return TrainBookingDB.getTrains();
    }

    @GET
    @Path("/trainid-{id}")
    public Train getTrain(@PathParam("id") String trainId){
        System.out.println("getTrain");
        Optional<Train> trainById = TrainBookingDB.getTrainById(trainId);
        if (trainById.isPresent())
            return trainById.get();
        else {
            throw new NotFoundException();
        }
    }

    @GET
    @Path("/search")
    public Response searchTrainsByCriteria(@QueryParam("departure") String departure,
                                           @QueryParam("arrival") String arrival,
                                           @QueryParam("departure_time") String departureTime) {
        return Response.ok().
                header("departure",departure).
                header("arrival",arrival).
                header("departure_time",departureTime).
                entity(TrainBookingDB.getTrains().subList(0,2)).build();
    }

    @Path("/bookings")
    public TrainBookingResource getTrainBookingResource(){
        System.out.println("TrainResource.getTrainBookingResource()");
        return new TrainBookingResource();
    }
}
