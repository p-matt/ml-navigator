var menu_index = 0;
var menu_possible_indexes = [0, 1, 2];

function main()
{
    button_next = document.getElementById("button-next");
    button_previous = document.getElementById("button-previous");
    
    layout_inputs = document.getElementById("container-inputs");
    layout_metrics = document.getElementById("container-metrics");
    layout_inferences = document.getElementById("container-inferences");

    let layouts_main = [layout_inputs, layout_metrics, layout_inferences];
    let layout_model_selection = document.getElementById("model-selected")
    button_next.onclick = function(){onClickMenuSelection(1, layouts_main, layout_model_selection)};
    button_previous.onclick = function(){onClickMenuSelection(-1, layouts_main, layout_model_selection)};
    
}

function onClickMenuSelection(index, layouts, layout_ms)
{
    tmp_index = menu_index + index;
    if (tmp_index < 0 || tmp_index > 2)
        return;

    menu_index += index;

    let layout_show = layouts[menu_index];

    hide_index = menu_possible_indexes.filter(v => v !== menu_index);
    layout_hide_1 = layouts[hide_index[0]];
    layout_hide_2 = layouts[hide_index[1]];
    layouts_hide = [layout_hide_1, layout_hide_2];

    x_coef = 100 * menu_index;
    x_target = -Math.abs(index * x_coef);

    DisplayButtons(menu_index);
    HideLayout(x_target, layouts_hide);
    ShowLayout(x_target, layout_show);
    DisplayModelSelected(layout_ms);
}

function DisplayButtons(idx)
{
    console.log(idx)
    if (idx === 0)
    {
        button_previous.style.visibility = "hidden";
        button_next.style.visibility = "unset";
        console.log("ici")
    }
    else if (idx === 1)
    {
        button_previous.style.visibility = "unset";
        button_next.style.visibility = "unset";
    }
    else
    {
        button_previous.style.visibility = "unset";
        button_next.style.visibility = "hidden";
    }
}


function HideLayout(x_target, layouts)
{
    layouts.forEach(layout =>
    {
        layout.style.transition = "transform .5s ease-out 0s";
        layout.style.transform = "translateX("+ x_target + "vw)";
    });
}

function ShowLayout(x_target, layout)
{
    layout.style.transition = "transform .5s ease-out 0s";
    layout.style.transform = "translateX(" + x_target + "vw)";
}

function DisplayModelSelected(layout_ms)
{
    if (menu_index == 1)
    {
        layout_ms.style.transition = "transform .5s ease-out 0s";
        layout_ms.style.transform = "translateY(" + -20 + "vh)";
        document.getElementById("container").style.height = "200vh"
    }
    else if (menu_index == 0)
    {
        layout_ms.style.transition = "transform .5s ease-out 0s";
        layout_ms.style.transform = "translateY(" + 0 + "vh)";
        document.getElementById("container").style.height = "100vh"
    }
}


var int_id = setInterval(function()
{
    if(document.readyState === "complete") 
    {
        if(document.getElementById("button-next") != null)
        {
            clearInterval(int_id);
            main(); 
        }
    }
}, 50);