using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class NewBehaviourScript : MonoBehaviour
{
    public string file;

    // Start is called before the first frame update
    void Start()
    {
        string path = string.Format("{0}/Resources/{1}.txt", Application.dataPath, file);
        StreamReader sr = File.OpenText(path);
        string str = null;
        Vector3 scale = new Vector3(0.1f, 0.1f, 0.1f);
        while ((str = sr.ReadLine()) != null)
        {
            string[] pointCoord = str.Split(' ');
            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.transform.position = new Vector3(float.Parse(pointCoord[0]), float.Parse(pointCoord[1]), float.Parse(pointCoord[2]));
            point.transform.localScale = scale;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
