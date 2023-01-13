using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class NewBehaviourScript : MonoBehaviour
{
    public string file;
    public bool hasMeta;

    // Start is called before the first frame update
    void Start()
    {
        string path = string.Format("{0}/Resources/{1}.txt", Application.dataPath, file);
        StreamReader sr = File.OpenText(path);
        string str = null;
        Vector3 scale = new Vector3(0.05f, 0.05f, 0.05f);
        if (hasMeta)
        {
            str = sr.ReadLine();
            string[] pointCoord = str.Split(' ');
            int raysO = int.Parse(pointCoord[0]), raysV = int.Parse(pointCoord[1]), vertices = int.Parse(pointCoord[2]);
            int H = int.Parse(pointCoord[3]), W = int.Parse(pointCoord[4]);
            List<GameObject> points = new List<GameObject>();
            while ((str = sr.ReadLine()) != null)
            {
                pointCoord = str.Split(' ');
                GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.transform.position = new Vector3(float.Parse(pointCoord[0]), float.Parse(pointCoord[1]), float.Parse(pointCoord[2]));
                point.transform.localScale = scale;
                points.Add(point);
            }
            Debug.DrawLine(points[0].transform.position, points[raysO].transform.position, Color.red, 10000);
            Debug.DrawLine(points[0].transform.position, points[raysO + W - 1].transform.position, Color.red, 10000);
            Debug.DrawLine(points[0].transform.position, points[raysO + raysV - 1].transform.position, Color.red, 10000);
            Debug.DrawLine(points[0].transform.position, points[raysO + raysV - W].transform.position, Color.red, 10000);
        }
        else
        {
            while ((str = sr.ReadLine()) != null)
            {
                string[] pointCoord = str.Split(' ');
                GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.transform.position = new Vector3(float.Parse(pointCoord[0]), float.Parse(pointCoord[1]), float.Parse(pointCoord[2]));
                point.transform.localScale = scale;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
