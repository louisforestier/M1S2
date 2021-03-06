package fr.mickaelbaron.jaxrstutorialexercice1;

import jakarta.ws.rs.core.UriBuilder;
import org.glassfish.jersey.logging.LoggingFeature;
import org.glassfish.grizzly.http.server.HttpServer;
import org.glassfish.jersey.grizzly2.httpserver.GrizzlyHttpServerFactory;
import org.glassfish.jersey.server.ResourceConfig;

import java.net.URI;
import java.util.logging.Level;


/**
 * @author Mickael BARON (baron.mickael@gmail.com)
 */
public class HelloLauncher {
	public static final URI BASE_URI = getBaseURI();

	private static URI getBaseURI(){
		return UriBuilder.fromUri("http://localhost/api").port(9991).build();
	}
	public static void main(String[] args) {
		ResourceConfig rc = new ResourceConfig();
		rc.registerClasses(HelloResource.class);
		rc.property(LoggingFeature.LOGGING_FEATURE_LOGGER_LEVEL_SERVER, Level.WARNING.getName());
		try {
			HttpServer server = GrizzlyHttpServerFactory.createHttpServer(BASE_URI,rc);
			server.start();
			System.out.println(String.format(
					"Jersey app started with WADL available at " + "%sapplication.wadl\nHit enter to stop it...",
					BASE_URI,BASE_URI));
			System.in.read();
			server.shutdown();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
