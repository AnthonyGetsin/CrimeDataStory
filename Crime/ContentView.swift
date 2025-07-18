import SwiftUI
import MapKit

// MARK: - Models

struct CrimeRecord: Codable, Identifiable {
    let id: String
    let incidentDate: String?
    let offense: String?
    let blockAddress: String?
    let location: CrimeLocation?
    
    enum CodingKeys: String, CodingKey {
        case id = "caseno"
        case incidentDate = "eventdt"
        case offense
        case blockAddress = "blkaddr"
        case location = "block_location"
    }
}

struct CrimeLocation: Codable {
    let latitude: String?
    let longitude: String?
    
    var coordinate: CLLocationCoordinate2D? {
        if let lat = latitude,
           let lon = longitude,
           let latDouble = Double(lat),
           let lonDouble = Double(lon) {
            return CLLocationCoordinate2D(latitude: latDouble, longitude: lonDouble)
        }
        return nil
    }
}

/// A helper model grouping crimes by street (block address)
struct CrimeStreet: Identifiable {
    let id: String      // using the blockAddress as an identifier
    let crimes: [CrimeRecord]
    let coordinate: CLLocationCoordinate2D
}

// MARK: - ViewModel

class CrimeDataViewModel: ObservableObject {
    @Published var records: [CrimeRecord] = []
    
    func fetchCrimeData() {
        // 1. Filter crimes from the past day.
        let dateFormatter = ISO8601DateFormatter()
        let endDate = Date()
        guard let startDate = Calendar.current.date(byAdding: .day, value: -1, to: endDate) else { return }
        let startDateString = dateFormatter.string(from: startDate)
        let endDateString = dateFormatter.string(from: endDate)
        
        // 2. Build the URL with the date filter. Increase the limit if needed.
        let urlString = "https://data.cityofberkeley.info/resource/k2nh-s5h5.json?$where=eventdt between '\(startDateString)' and '\(endDateString)'&$limit=100"
        
        guard let url = URL(string: urlString) else {
            print("Invalid URL")
            return
        }
        
        var request = URLRequest(url: url)
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        
        // 3. Fetch and decode the JSON.
        URLSession.shared.dataTask(with: request) { data, _, error in
            guard let data = data, error == nil else {
                print("Error fetching data: \(error?.localizedDescription ?? "Unknown error")")
                return
            }
            
            do {
                let decoder = JSONDecoder()
                // Try decoding as an array first
                if let recordsArray = try? decoder.decode([CrimeRecord].self, from: data) {
                    DispatchQueue.main.async {
                        self.records = recordsArray
                    }
                } else {
                    // If the API returns a dictionary (perhaps with a "data" key),
                    // decode that key instead.
                    let wrapper = try decoder.decode([String: [CrimeRecord]].self, from: data)
                    if let recordsArray = wrapper["data"] {
                        DispatchQueue.main.async {
                            self.records = recordsArray
                        }
                    } else {
                        print("Unexpected JSON structure")
                    }
                }
            } catch {
                print("Decoding error: \(error)")
            }
        }.resume()
    }
    
    /// Groups the fetched crime records by street (blockAddress)
    var groupedStreets: [CrimeStreet] {
        let grouped = Dictionary(grouping: records, by: { $0.blockAddress ?? "Unknown" })
        return grouped.compactMap { (key, crimes) in
            // Use only records that have valid coordinates.
            let coordinates = crimes.compactMap { $0.location?.coordinate }
            guard !coordinates.isEmpty else { return nil }
            // Calculate an average coordinate for this street.
            let avgLatitude = coordinates.map { $0.latitude }.reduce(0, +) / Double(coordinates.count)
            let avgLongitude = coordinates.map { $0.longitude }.reduce(0, +) / Double(coordinates.count)
            return CrimeStreet(id: key, crimes: crimes, coordinate: CLLocationCoordinate2D(latitude: avgLatitude, longitude: avgLongitude))
        }
    }
}

// MARK: - Views

struct ContentView: View {
    @StateObject var viewModel = CrimeDataViewModel()
    
    @State private var region = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 37.8715, longitude: -122.2730),
        span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
    )
    
    @State private var selectedStreet: CrimeStreet?
    @State private var showCrimeList = false
    
    var body: some View {
        VStack {
            // Map displays grouped street overlays rather than individual pins.
            Map(coordinateRegion: $region, annotationItems: viewModel.groupedStreets) { street in
                MapAnnotation(coordinate: street.coordinate) {
                    // A circle to represent the street's crime density.
                    Circle()
                        .fill(colorForCrimeCount(count: street.crimes.count))
                        .frame(width: 30, height: 30)
                        .overlay(
                            // Display the number of crimes as text within the circle.
                            Text("\(street.crimes.count)")
                                .font(.caption)
                                .foregroundColor(.black)
                        )
                        .onTapGesture {
                            // Selecting a street opens the list of crimes for that street.
                            selectedStreet = street
                            showCrimeList = true
                        }
                }
            }
            .edgesIgnoringSafeArea(.all)
            .onAppear {
                viewModel.fetchCrimeData()
            }
            
            // Zoom In/Out Controls
            HStack {
                Button(action: {
                    region.span.latitudeDelta /= 2
                    region.span.longitudeDelta /= 2
                }) {
                    Image(systemName: "plus.magnifyingglass")
                        .padding()
                }
                Button(action: {
                    region.span.latitudeDelta *= 2
                    region.span.longitudeDelta *= 2
                }) {
                    Image(systemName: "minus.magnifyingglass")
                        .padding()
                }
            }
        }
        // Show a sheet with a list of crimes when a street is tapped.
        .sheet(isPresented: $showCrimeList) {
            if let street = selectedStreet {
                CrimeListView(street: street)
            }
        }
    }
    
    // Color the circle based on the count of crimes:
    // • Green: 1–2 crimes (low)
    // • Yellow: 3–5 crimes (moderate)
    // • Red: More than 5 crimes (high)
    func colorForCrimeCount(count: Int) -> Color {
        if count <= 2 {
            return .green
        } else if count <= 5 {
            return .yellow
        } else {
            return .red
        }
    }
}

/// Displays a list of crimes grouped by a given street.
struct CrimeListView: View {
    let street: CrimeStreet
    
    var body: some View {
        NavigationView {
            List(street.crimes.sorted(by: { ($0.incidentDate ?? "") > ($1.incidentDate ?? "") })) { crime in
                VStack(alignment: .leading) {
                    Text(crime.offense ?? "Unknown Offense")
                        .font(.headline)
                    Text(crime.incidentDate ?? "Unknown Date")
                        .font(.subheadline)
                }
            }
            .navigationTitle(street.id)
        }
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
