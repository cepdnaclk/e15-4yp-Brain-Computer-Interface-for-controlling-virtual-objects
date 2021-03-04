using System;
using UnityEngine;
using UnityEngine.UI;

public class SphereController : MonoBehaviour{

    public float speed = 1f;
    public Button resetButton;
    // Update is called once per frame
    private void Start() {
        Button btn = resetButton.GetComponent<Button>();
        btn.onClick.AddListener(TaskOnClick);
    }

    private void TaskOnClick() {
        Vector3 initPosition = new Vector3(0f,1.91f,-8.88f);
        transform.localPosition = initPosition;
    }


    void Update() {
        if (Input.GetKey("left")) {    //move object to left
            Vector3 vect = new Vector3(-speed * Time.deltaTime, 0f, 0f);
            transform.localPosition += transform.localRotation * vect;
        }else if (Input.GetKey("right")) {    //move object to right side
            Vector3 vect = new Vector3(speed * Time.deltaTime, 0f, 0f);
            transform.localPosition += transform.localRotation * vect;
        }else if (Input.GetKey("w")) {    //move object to forward
            Vector3 vect = new Vector3(0f, 0f, speed * Time.deltaTime);
            transform.localPosition += transform.localRotation * vect;
        }else if (Input.GetKey("s")) {
            Vector3 vect = new Vector3(0f, 0f, -speed * Time.deltaTime);
            transform.localPosition += transform.localRotation * vect;
        }
    }
    
    private void OnTriggerEnter(Collider other){
        print("Sphere collided");
        string time = DateTime.Now.ToString("h:mm:ss tt");    //collided time
    }
}
