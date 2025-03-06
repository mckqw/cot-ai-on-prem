import UIKit
import Alamofire

class ViewController: UIViewController {
    // Outlets for UI elements
    @IBOutlet weak var queryTextField: UITextField!
    @IBOutlet weak var responseTextView: UITextView!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!

    override func viewDidLoad() {
        super.viewDidLoad()
        // Hide the activity indicator when not in use
        activityIndicator.hidesWhenStopped = true
        // Make the text view non-editable
        responseTextView.isEditable = false
    }

    // Action triggered when the submit button is pressed
    @IBAction func submitQuery(_ sender: UIButton) {
        guard let query = queryTextField.text, !query.isEmpty else {
            showAlert(message: "Please enter a query.")
            return
        }
        activityIndicator.startAnimating()
        sendQueryToServer(query: query)
    }

    // Function to send the query to the server
    func sendQueryToServer(query: String) {
        let parameters: [String: String] = ["text": query]
        // Replace "server-ip" with your actual server IP address
        AF.request("http://server-ip:8000/query", 
                   method: .post, 
                   parameters: parameters, 
                   encoding: JSONEncoding.default)
            .responseJSON { response in
                self.activityIndicator.stopAnimating()
                switch response.result {
                case .success(let value):
                    if let json = value as? [String: Any], let answer = json["answer"] as? String {
                        self.responseTextView.text = answer
                    } else {
                        self.showAlert(message: "Invalid response from server.")
                    }
                case .failure(let error):
                    self.showAlert(message: "Network error: \(error.localizedDescription)")
                }
            }
    }

    // Helper function to display error alerts
    func showAlert(message: String) {
        let alert = UIAlertController(title: "Error", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}